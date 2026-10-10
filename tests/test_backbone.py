"""Offline contract with genuine Gemma4 and tiny random weights, no download."""

import torch
from transformers import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    AutoModelForMultimodalLM,
)
from zeos_vla.backbone import FrozenGemma
from zeos_vla.head import FlowHead, HeadConfig


def test_genuine_gemma_multimodal_final_states_are_frozen():
    torch.set_num_threads(1)
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=64,
    )
    text = Gemma4TextConfig(
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
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    cfg = Gemma4Config(
        text_config=text, vision_config=vision, audio_config=None, image_token_id=63
    )
    model = AutoModelForMultimodalLM.from_config(cfg).eval().requires_grad_(False)
    ids = torch.tensor([[1, 63, 3, 4]])
    inputs = {
        "input_ids": ids,
        "pixel_values": torch.randn(1, 4, 12),
        "image_position_ids": torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]),
    }
    wrapper = FrozenGemma.__new__(FrozenGemma)
    wrapper.model = model
    wrapper.inputs = lambda image, instruction: inputs
    features, mask = wrapper.extract(None, "test")
    assert features.shape == (1, 4, 32) and torch.isfinite(features).all()
    assert not features.requires_grad and all(
        not p.requires_grad for p in model.parameters()
    )
    head = FlowHead(HeadConfig(feature_dim=32, width=32, layers=1))
    loss = head.loss(
        torch.randn(1, 16, 7), features.clone(), torch.randn(1, 7), mask.clone()
    )
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert head.feature[1].weight.grad is not None
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    assert output.shape[1] > ids.shape[1]
