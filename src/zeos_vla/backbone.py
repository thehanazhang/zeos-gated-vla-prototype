"""Frozen Gemma: original multimodal prompt features and semantic JSON generation."""

from dataclasses import dataclass
import time
import torch
from PIL import Image

MODEL_ID = "google/gemma-3-4b-it"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
PROMPT_VERSION = "package-v1"
SYSTEM = """You control a simulated robot. Return ONLY one JSON object, no markdown.
Schema: {"action":"manipulate","target":"package","speed":"slow" or "fast"}.
Use this schema for package or Maltesers manipulation. Choose the requested speed. Do not decide safety permission;
a separate runtime gate will assess your proposal. No motion occurs from this text."""


@dataclass
class ObservationEncoding:
    features: torch.Tensor
    mask: torch.Tensor
    command_text: str
    metrics: dict


class FrozenGemma:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        import os
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        if "HF_TOKEN" not in os.environ and "HUGGINGFACE_TOKEN" in os.environ:
            os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_dim = self.model.config.text_config.hidden_size

    def inputs(self, image: Image.Image, instruction: str, feedback: str = ""):
        text = (
            instruction
            if not feedback
            else instruction + "\nRuntime feedback: " + feedback
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": text},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.inference_mode()
    def extract(self, image, instruction):
        inputs = self.inputs(image, instruction)
        features = self.model.model(
            **inputs, use_cache=False, return_dict=True
        ).last_hidden_state
        return features.detach().clone(), inputs["attention_mask"].bool()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(self, image, instruction, max_new_tokens=96):
        inputs = self.inputs(image, instruction)
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        # The base model exposes its final layer directly. No hooks, decoder training,
        # KV-cache extension, or all-layer hidden-state retention is needed.
        features = self.model.model(
            **inputs, use_cache=False, return_dict=True
        ).last_hidden_state
        features = features.detach().clone()
        self._sync()
        prefill = time.perf_counter() - start
        start = time.perf_counter()
        # Native Transformers generation cache only; no custom cache access/mutation.
        ids = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
        self._sync()
        generation = time.perf_counter() - start
        text = self.processor.decode(
            ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        metrics = {
            "prefill_seconds": prefill,
            "generation_seconds": generation,
            "prompt_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": ids.shape[1] - inputs["input_ids"].shape[1],
            "feature_shape": list(features.shape),
            "backbone_frozen": all(
                not p.requires_grad for p in self.model.parameters()
            ),
        }
        if self.device.type == "cuda":
            metrics.update(
                gpu=torch.cuda.get_device_name(self.device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            )
        return ObservationEncoding(
            features, inputs["attention_mask"].bool(), text, metrics
        )

    @torch.inference_mode()
    def feedback(self, image, instruction, reason):
        inputs = self.inputs(image, instruction, reason)
        ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=96)
        return self.processor.decode(
            ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
