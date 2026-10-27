"""
pos_local_config.py -- write PoS configs (github.com/luoyu100/PoS, MIT) that use a LOCAL OpenAI-compatible server (vLLM) instead of
the DashScope API, so RCA100 data never leaves the machine. Writes <PoS>/configs/rca100/{pos,raw}_local.yaml.

    python scripts/pos_local_config.py --pos /kaggle/working/PoS --model qwen-local --base-url http://127.0.0.1:8000/v1 \
        --start 0 --end 1 --run-name qwen25-7b-local

Every model role (policy, belief, sentinel) points to the same local model, as in the PoS paper's same-backbone setting.
"""
import argparse
import os

import yaml


def patch_model(block, model, base_url):
    block["model"] = model
    block["base_url"] = base_url
    block["api_key_env"] = "LOCAL_LLM_KEY"           # vLLM ignores the key; PoS only needs the variable to exist
    eb = block.get("extra_body") or {}
    eb.pop("enable_thinking", None)                 # DashScope field; vLLM takes chat-template kwargs instead
    eb["chat_template_kwargs"] = {"enable_thinking": False}
    block["extra_body"] = eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--model", default="qwen-local")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=103)
    ap.add_argument("--run-name", default="local")
    a = ap.parse_args()
    for name in ("pos", "raw"):
        src = os.path.join(a.pos, "configs", "rca100", f"{name}.yaml")
        cfg = yaml.safe_load(open(src))
        patch_model(cfg, a.model, a.base_url)
        ctx = cfg.get("context") or {}
        if isinstance(ctx.get("model"), dict):
            patch_model(ctx["model"], a.model, a.base_url)
        if isinstance((ctx.get("sentinel") or {}).get("model"), dict):
            patch_model(ctx["sentinel"]["model"], a.model, a.base_url)
        cfg["start_case"], cfg["end_case"] = a.start, a.end
        cfg["output_run_name"] = f"{a.run_name}-{name}"
        dst = os.path.join(a.pos, "configs", "rca100", f"{name}_local.yaml")
        yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
        print("written", dst)


if __name__ == "__main__":
    main()
