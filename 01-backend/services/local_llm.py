"""Local LLM inference for GPU-accelerated processing.

Uses transformers with 4-bit quantization for 24GB VRAM GPUs (L4, RTX 3090/4090).
"""

from __future__ import annotations

import torch
from loguru import logger
from pydantic import BaseModel, Field


class LocalLLMConfig(BaseModel):
    """Configuration for local LLM."""

    model_name: str = Field(
        default="Qwen/Qwen2.5-32B-Instruct",
        description="HuggingFace model name",
    )
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    device: str = Field(default="cuda")
    load_in_4bit: bool = Field(default=True)  # Key for 24GB VRAM


class LocalLLM:
    """Local transformer-based LLM with GPU acceleration."""

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()
        self._pipeline = None
        self._tokenizer = None

    def _load(self):
        """Lazy load the model."""
        if self._pipeline is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading {self.config.model_name}...")

            # 4-bit quantization config
            quantization_config = None
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit quantization (~18GB VRAM for 32B model)")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            # Create pipeline
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                device_map="auto",
            )

            logger.info(f"Model loaded on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def complete(
        self,
        system_prompt: str | None,
        user_prompt: str,
    ) -> str:
        """Generate completion."""
        self._load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Format prompt
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.debug(f"Generating with prompt length: {len(prompt)}")

        # Generate
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            return_full_text=False,
        )

        return outputs[0]["generated_text"]

    def get_gpu_memory(self) -> dict:
        """Check GPU memory usage."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


class LLMResponse:
    """Standardized LLM response for compatibility."""

    def __init__(self, content: str, model: str = "local", usage: dict | None = None):
        self.content = content
        self.model = model
        self.usage = usage or {}


class LocalLLMWrapper:
    """Wraps LocalLLM to match LLMClient interface."""

    def __init__(self, local_llm: LocalLLM):
        self.local_llm = local_llm

    async def complete(
        self,
        system_prompt: str | None,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Async wrapper around local LLM complete."""
        import asyncio

        # Run sync method in thread pool
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: self.local_llm.complete(system_prompt, user_prompt)
        )

        return LLMResponse(content=content, model=self.local_llm.config.model_name)
