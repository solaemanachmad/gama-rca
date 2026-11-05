"""
openai_compat_server.py -- minimal OpenAI-compatible chat server for a LOCAL transformers model (no vLLM, no external API).
Used to run PoS (github.com/luoyu100/PoS) with the same local backbone as our systems on Kaggle T4 GPUs, where recent vLLM builds do not
install cleanly. Standard library HTTP server; one generation at a time (a lock); greedy decoding when temperature == 0.

    python scripts/openai_compat_server.py --model /kaggle/input/models/.../7b-instruct/1 --port 8000 [--max-input-tokens 24000]
    python scripts/openai_compat_server.py --fake --port 8000          # plumbing test without a model

Endpoints: GET /v1/models, POST /v1/chat/completions (messages, max_tokens, temperature, response_format, chat_template_kwargs).
With response_format {"type": "json_object"} the first balanced {...} of the output is returned when the model adds text around it.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"gen": None, "name": "qwen-local", "calls": 0, "max_in": 24000}


def first_json(text: str) -> str:
    start = text.find("{")
    while start != -1:
        depth, ins, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if ins:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    ins = False
                continue
            if ch == '"':
                ins = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return text


def load_transformers(path, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map="auto")
    model.eval()

    def gen(messages, max_tokens, temperature, template_kwargs):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))
        ids = tok(text, return_tensors="pt")["input_ids"]
        n_in = ids.shape[-1]
        if n_in > STATE["max_in"]:                         # keep head (instructions) and tail (latest turn), as agents/kaggle_client.py
            head = int(STATE["max_in"] * 0.4)
            ids = torch.cat([ids[0, :head], ids[0, -(STATE["max_in"] - head):]]).unsqueeze(0)
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                 do_sample=temperature > 0, temperature=max(temperature, 1e-4), pad_token_id=tok.eos_token_id)
        new = out[0][ids.shape[-1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), int(ids.shape[-1]), int(len(new))
    return gen


def fake_gen(messages, max_tokens, temperature, template_kwargs):
    return 'Sure: {"action": "noop", "note": "fake"} done', 10, 5


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": STATE["name"], "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            msgs = req.get("messages") or []
            mt = int(req.get("max_tokens") or req.get("max_completion_tokens") or 1024)
            temp = float(req.get("temperature") or 0.0)
            with LOCK:
                t0 = time.time()
                text, n_in, n_out = STATE["gen"](msgs, mt, temp, req.get("chat_template_kwargs"))
                STATE["calls"] += 1
            if (req.get("response_format") or {}).get("type") == "json_object":
                text = first_json(text)
            self._send(200, {"id": f"local-{STATE['calls']}", "object": "chat.completion", "created": int(time.time()),
                             "model": STATE["name"], "choices": [{"index": 0, "finish_reason": "stop",
                                                                  "message": {"role": "assistant", "content": text}}],
                             "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
                                       "latency_s": round(time.time() - t0, 2)}})
        except Exception as e:
            self._send(500, {"error": {"message": repr(e)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--name", default="qwen-local")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-input-tokens", type=int, default=24000)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    STATE["name"], STATE["max_in"] = a.name, a.max_input_tokens
    STATE["gen"] = fake_gen if a.fake else load_transformers(a.model, a.dtype)
    print(f"serving {a.name} on :{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
