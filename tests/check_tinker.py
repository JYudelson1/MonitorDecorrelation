"""Cheapest tinker derisk: confirm .env loads, auth works, and Qwen3-8B is available.

Run: uv run python experiments/check_tinker.py
"""

from __future__ import annotations

from dotenv import load_dotenv

import tinker

load_dotenv()


def main() -> None:
    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    models = [m.model_name for m in caps.supported_models]
    print(f"auth OK — {len(models)} supported models")
    target = "Qwen/Qwen3-8B"
    print(f"{target} available: {target in models}")
    for m in models:
        print("  ", m)


if __name__ == "__main__":
    main()
