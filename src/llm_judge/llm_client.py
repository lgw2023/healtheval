from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import urllib.request

from .llm_scorer import LLMCaller


@dataclass
class LLMEnvConfig:
    url: str
    model: str
    api_key: str

    @classmethod
    def from_env(cls) -> "LLMEnvConfig":
        url = os.getenv("LLM_MODEL_URL")
        model = os.getenv("LLM_MODEL_NAME")
        api_key = os.getenv("LLM_MODEL_API_KEY")
        if not all([url, model, api_key]):
            missing = [
                name
                for name, value in [
                    ("LLM_MODEL_URL", url),
                    ("LLM_MODEL_NAME", model),
                    ("LLM_MODEL_API_KEY", api_key),
                ]
                if not value
            ]
            raise EnvironmentError(
                f"Missing environment variables for LLM caller: {', '.join(missing)}"
            )
        return cls(url=url, model=model, api_key=api_key)


class OpenAILikeCaller:
    """Minimal caller compatible with OpenAI-style chat completion APIs."""

    def __init__(self, config: LLMEnvConfig, timeout: int = 120):
        self.config = config
        self.timeout = timeout

    def __call__(
        self, prompt: str, temperature: float, top_k: int | None, top_p: float | None, seed: int
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only judge. Return only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        # Some providers support deterministic seeds; send it when available.
        payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            self.config.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            resp_text = response.read().decode("utf-8")
        try:
            data = json.loads(resp_text)
        except json.JSONDecodeError:
            return resp_text
        if isinstance(data, dict) and data.get("choices"):
            first_choice = data["choices"][0]
            # OpenAI responses: choices[].message.content
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else first_choice.get("text")
                if content:
                    return content
        # Fallback to raw text
        if isinstance(data, dict) and "text" in data:
            return str(data["text"])
        return resp_text


class MockLLMCaller:
    """Offline fallback when no model endpoint is available."""

    def __init__(self, rules: Optional[int] = None):
        self.rules = rules or 6

    def __call__(
        self, prompt: str, temperature: float, top_k: int | None, top_p: float | None, seed: int
    ) -> str:
        rng_seed = int(hashlib.md5(f"{prompt}|{seed}".encode("utf-8")).hexdigest(), 16)
        rng = random.Random(rng_seed)
        # Make the number of hits dependent on sampling params to mimic variability.
        variability = rng.randint(0, self.rules)
        checks = [
            {
                "rule_id": f"MOCK_RULE_{idx}",
                "hit": idx < variability,
                "severity": "strict",
                "reason": "mock",
                "excerpt": "",
            }
            for idx in range(self.rules)
        ]
        confidence = max(0.1, min(1.0, 1 - abs(temperature - 0.2)))
        payload = {"checks": checks, "confidence": confidence}
        return json.dumps(payload, ensure_ascii=False)


def build_llm_caller(mock: bool = False, timeout: int = 120) -> LLMCaller:
    if mock:
        return MockLLMCaller()
    config = LLMEnvConfig.from_env()
    return OpenAILikeCaller(config, timeout=timeout)

