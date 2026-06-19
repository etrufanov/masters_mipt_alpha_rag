"""Интерфейс LLM и две реализации: FakeLLM (для тестов) и VLLMModel (боевая, на GPU)."""
from typing import Protocol


class LLM(Protocol):
    """Протокол реализации языковой модели."""

    def generate(
        self, prompts: list[str], max_tokens: int, temperature: float, stop: list[str] | None
    ) -> list[str]:
        """Сгенерировать продолжения для списка промптов."""
        ...


class FakeLLM:
    """Детерминированная заглушка для тестов (без GPU)."""

    def __init__(self, outputs: list[str] | None = None):
        self._outputs = outputs

    def generate(
        self, prompts: list[str], max_tokens: int, temperature: float, stop: list[str] | None
    ) -> list[str]:
        """Вернуть заранее заданные ответы либо ECHO на каждый промпт."""
        if self._outputs is None:
            return ["ECHO" for _ in prompts]
        assert len(self._outputs) >= len(prompts), (
            f"FakeLLM: {len(self._outputs)} scripted outputs < {len(prompts)} prompts")
        return list(self._outputs[:len(prompts)])


class VLLMModel:
    """Боевая модель через vLLM. vLLM установлен только на GPU (extra `gpu`)."""

    def __init__(
        self,
        model: str,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM as _VLLM, SamplingParams
        from transformers import AutoTokenizer

        self._SamplingParams = SamplingParams
        self._tok = AutoTokenizer.from_pretrained(model)
        self._llm = _VLLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(
        self, prompts: list[str], max_tokens: int, temperature: float, stop: list[str] | None
    ) -> list[str]:
        """Сгенерировать ответы через vLLM, применив chat-template модели к каждому промпту."""
        sp = self._SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop)
        texts = [
            self._tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
        outs = self._llm.generate(texts, sp)
        return [o.outputs[0].text.strip() for o in outs]
