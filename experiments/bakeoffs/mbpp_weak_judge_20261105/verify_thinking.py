"""The VLLM_JUDGES admission check (monitors/vllm.py module doc) for a served model:
thinking on -> message.reasoning + clean content; off -> no reasoning; thinking_token_budget N -> ~N reasoning tokens.
  python verify_thinking.py <model> <base_url>"""
import sys, httpx, json
model, base = sys.argv[1], sys.argv[2].rstrip("/")
P = "Is 1001 prime? Think it through, then answer with exactly one line: `SCORE: <0-100>` (100 = certainly prime)."
def call(**kw):
    body = {"model": model, "messages": [{"role": "user", "content": P}], "temperature": 1.0, "max_tokens": 2048, **kw}
    d = httpx.post(f"{base}/chat/completions", json=body, timeout=600).json()
    m = d["choices"][0]["message"]; u = d.get("usage", {})
    return m, d["choices"][0]["finish_reason"], (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), u.get("completion_tokens")
ok = True
m, f, rt, ct = call(chat_template_kwargs={"enable_thinking": True})
r = m.get("reasoning") or m.get("reasoning_content")
print(f"ON : finish={f} reasoning_tokens={rt} completion={ct} reasoning_len={len(r or '')} content={m.get('content')!r:.120}")
ok &= bool(r) and "</think>" not in (m.get("content") or "") and "<think>" not in (m.get("content") or "")
m, f, rt, ct = call(chat_template_kwargs={"enable_thinking": False})
r = m.get("reasoning") or m.get("reasoning_content")
print(f"OFF: finish={f} reasoning_tokens={rt} reasoning_len={len(r or '')} content={m.get('content')!r:.120}")
ok &= not r
m, f, rt, ct = call(chat_template_kwargs={"enable_thinking": True}, thinking_token_budget=64)
r = m.get("reasoning") or m.get("reasoning_content")
print(f"BUDGET64: finish={f} reasoning_tokens={rt} completion={ct} content={m.get('content')!r:.160}")
ok &= rt is not None and rt <= 70 and "</think>" not in (m.get("content") or "")
print("VERDICT:", "PASS" if ok else "FAIL")
