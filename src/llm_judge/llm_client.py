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

        # 如果存在代理相关环境变量，则显式为 urllib 配置 ProxyHandler，
        # 确保所有模型调用都严格走代理（例如本地 Clash / Surge 等）。
        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        proxies: Dict[str, str] = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy

        if proxies:
            proxy_handler = urllib.request.ProxyHandler(proxies)
            self._opener = urllib.request.build_opener(proxy_handler)
        else:
            # 不配置代理时使用默认 opener
            self._opener = urllib.request.build_opener()

    @property
    def _endpoint(self) -> str:
        """Normalize endpoint to support both base-url and full-path configs.

        DeepInfra 提供的 OpenAI 兼容接口通常是::

            https://api.deepinfra.com/v1/openai/chat/completions

        但本项目通过环境变量传入的可能只是 ``https://api.deepinfra.com/v1/openai``。
        为了兼容两种写法，这里做一个轻量级归一化：

        - 如果 URL 中已经包含 ``chat/completions``，则原样使用；
        - 否则在末尾补上 ``/chat/completions``。
        """

        base = self.config.url.rstrip("/")
        if "chat/completions" in base:
            return base
        return f"{base}/chat/completions"

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
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        # Some providers support deterministic seeds; send it when available.
        payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        # 统一通过初始化时构造的 opener 发起请求，以便自动应用代理设置
        with self._opener.open(request, timeout=self.timeout) as response:
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

