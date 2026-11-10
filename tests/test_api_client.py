"""ApiChatClient against a fake HTTP session (no network, no key): request body, retry on 429 with retry-after, usage counts,
reasoning kept out of the answer, the key never in repr / stats.

    python tests/test_api_client.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import agents.api_client as ac   # noqa: E402


class Resp:
    def __init__(self, code, js=None, headers=None, text=""):
        self.status_code, self._js, self.headers, self.text = code, js, headers or {}, text

    def json(self):
        return self._js


class Session:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return self.script.pop(0)


OK = {"choices": [{"finish_reason": "stop", "message": {"content": ' {"predicted_fault_type": "F034-cpuFullLoad"} ',
                                                         "reasoning": "cpu rises first..."}}],
      "usage": {"prompt_tokens": 900, "completion_tokens": 120, "completion_tokens_details": {"reasoning_tokens": 80}}}


def main():
    ac.time.sleep = lambda s: None
    s = Session([Resp(429, headers={"retry-after": "3"}), Resp(503), Resp(200, OK)])
    c = ac.ApiChatClient(base_url="https://x.test/v1", model="m", api_key="SECRET-KEY", temperature=0.0, max_tokens=8192,
                         reasoning_effort="medium", session=s)
    out = c.generate_json("prompt", system="sys")
    assert out == {"predicted_fault_type": "F034-cpuFullLoad"}, out
    url, body, headers = s.calls[-1]
    assert url == "https://x.test/v1/chat/completions" and len(s.calls) == 3
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    assert body["temperature"] == 0.0 and body["seed"] == 0 and body["max_completion_tokens"] == 8192 and body["reasoning_effort"] == "medium"
    st = c.usage_stats()
    assert (st["total_calls"], st["input_tokens"], st["output_tokens"], st["reasoning_tokens"], st["api_retries"],
            st["api_rate_limited"]) == (1, 900, 120, 80, 2, 1), st
    assert "SECRET-KEY" not in repr(c) and "SECRET-KEY" not in str(st)
    s2 = Session([Resp(400, text="bad request")])
    c2 = ac.ApiChatClient(base_url="https://x.test/v1", model="m", api_key="K", session=s2)
    try:
        c2.generate("p")
        raise AssertionError("no error raised")
    except RuntimeError as e:
        assert "400" in str(e) and "K" not in str(e).replace("400", "").replace("bad request", "").replace("API error", "")
    print("3/3 passed")


def test_suite():
    main()


if __name__ == "__main__":
    main()
