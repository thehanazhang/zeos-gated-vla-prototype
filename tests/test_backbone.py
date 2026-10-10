"""Offline contract test with genuine Transformers Gemma3 (tiny random weights).
This is an implementation contract test, not the pretrained-4B verification.
"""

import torch
from transformers import (
    Gemma3Config,
    Gemma3TextConfig,
    SiglipVisionConfig,
    Gemma3ForConditionalGeneration,
)
from zeos_vla.backbone import FrozenGemma
from zeos_vla.head import FlowHead, HeadConfig


def test_genuine_gemma_multimodal_final_states_are_frozen():
    torch.set_num_threads(1)
    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    )
    text = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
        pad_token_id=0,
        eos_token_id=2,
    )
    cfg = Gemma3Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        mm_tokens_per_image=4,
        image_token_index=63,
        boi_token_index=61,
        eoi_token_index=62,
    )
    model = Gemma3ForConditionalGeneration(cfg).eval().requires_grad_(False)
    ids = torch.tensor([[1, 63, 63, 63, 63, 3, 4]])
    inputs = {
        "input_ids": ids,
        "pixel_values": torch.randn(1, 3, 28, 28),
        "attention_mask": torch.ones_like(ids),
        "token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0, 0]]),
    }
    wrapper = FrozenGemma.__new__(FrozenGemma)
    wrapper.model = model
    wrapper.inputs = lambda image, instruction: inputs
    features, mask = wrapper.extract(None, "test")
    assert features.shape == (1, 7, 32) and torch.isfinite(features).all()
    assert not features.requires_grad and all(
        not p.requires_grad for p in model.parameters()
    )
    head = FlowHead(HeadConfig(feature_dim=32, width=32, layers=1))
    # Cache serialization/cloning exits inference-tensor context before autograd.
    loss = head.loss(
        torch.randn(1, 16, 7), features.clone(), torch.randn(1, 7), mask.clone()
    )
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert head.feature[1].weight.grad is not None
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    assert output.shape[1] > ids.shape[1]
