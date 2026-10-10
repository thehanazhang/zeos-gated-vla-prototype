import torch
import pytest
from zeos_vla.head import FlowHead, HeadConfig


def small():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    return FlowHead(HeadConfig(feature_dim=24, width=32, layers=1))


def batch():
    return (
        torch.randn(2, 16, 7),
        torch.randn(2, 9, 24),
        torch.randn(2, 7),
        torch.ones(2, 9, dtype=torch.bool),
    )


def test_dimensions_loss_gradients_and_sampling():
    m = small()
    a, f, p, mask = batch()
    f.requires_grad_(False)
    loss = m.loss(a, f, p, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()
    )
    assert m.feature[1].weight.grad.abs().sum() > 0
    assert m.transformer.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    assert f.grad is None
    assert m.sample(f, p, mask, steps=4).shape == (2, 16, 7)


def test_masks_and_invalid_shapes():
    m = small()
    a, f, p, mask = batch()
    with pytest.raises(ValueError):
        m.sample(f, p, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        m.loss(a, f, p, mask, torch.zeros(2, 16, dtype=torch.bool))
    with pytest.raises(ValueError):
        m(a[:, :8], torch.rand(2), f, p, mask)


def test_can_fit_a_fixed_synthetic_flow_batch():
    m = small()
    a, f, p, mask = batch()
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    values = []
    for _ in range(60):
        opt.zero_grad()
        loss = m.loss(a, f, p, mask, generator=torch.Generator().manual_seed(2))
        loss.backward()
        opt.step()
        values.append(loss.item())
    assert values[-1] < values[0] * 0.25


def test_padding_does_not_change_predictions():
    m = small().eval()
    a, f, p, mask = batch()
    t = torch.rand(2)
    with torch.no_grad():
        base = m(a, t, f, p, mask)
        extended = m(
            a,
            t,
            torch.cat((f, torch.randn(2, 3, 24)), 1),
            p,
            torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), 1),
        )
    torch.testing.assert_close(base, extended, atol=1e-5, rtol=1e-5)
