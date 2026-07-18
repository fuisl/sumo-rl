from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reference_diffusion_graph_conv(conv, inputs, state, torch):
    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size, conv.num_nodes, -1)
    state = state.reshape(batch_size, conv.num_nodes, -1)
    x = torch.cat([inputs, state], dim=-1)
    x = x.permute(1, 2, 0).reshape(conv.num_nodes, -1)

    diffusion_terms = [x]
    for support in conv.supports:
        x_k = x
        for _ in range(conv.max_diffusion_step):
            x_k = torch.matmul(support.to(device=x.device, dtype=x.dtype), x_k)
            diffusion_terms.append(x_k)

    x = torch.stack(diffusion_terms, dim=0)
    x = x.reshape(conv.num_matrices, conv.num_nodes, conv.input_size, batch_size)
    x = x.permute(3, 1, 2, 0).reshape(batch_size * conv.num_nodes, conv.input_size * conv.num_matrices)
    x = torch.matmul(x, conv.weight) + conv.bias
    return x.reshape(batch_size, -1)


def test_diffusion_graph_conv_matches_reference_tensorized_path():
    torch = pytest.importorskip("torch")
    from sumo_rl.models.dcrnn import DiffusionGraphConv

    for seed in range(4):
        torch.manual_seed(seed)
        num_nodes = 3 + seed
        input_dim = 4
        hidden_dim = 5
        max_diffusion_step = 1 + (seed % 3)
        output_dim = 7
        rng = np.random.RandomState(seed)
        supports = [
            rng.rand(num_nodes, num_nodes).astype(np.float32),
            np.random.RandomState(seed + 100).rand(num_nodes, num_nodes).astype(np.float32),
        ]
        conv = DiffusionGraphConv(
            supports=supports,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=output_dim,
        )

        for batch_size in (1, 2, 5):
            inputs = torch.randn(batch_size, num_nodes * input_dim, dtype=torch.float32)
            state = torch.randn(batch_size, num_nodes * hidden_dim, dtype=torch.float32)

            with torch.no_grad():
                expected = _reference_diffusion_graph_conv(conv, inputs, state, torch)
                actual = conv(inputs, state)

            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_diffusion_graph_conv_matches_reference_gradients():
    torch = pytest.importorskip("torch")
    from sumo_rl.models.dcrnn import DiffusionGraphConv

    adjacency = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.25],
            [0.0, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    conv = DiffusionGraphConv(
        supports=[adjacency],
        input_dim=3,
        hidden_dim=2,
        num_nodes=3,
        max_diffusion_step=2,
        output_dim=4,
    )
    inputs = torch.randn(2, 9, dtype=torch.float32, requires_grad=True)
    state = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)

    expected = _reference_diffusion_graph_conv(conv, inputs, state, torch).sum()
    expected.backward()
    expected_input_grad = inputs.grad.detach().clone()
    expected_state_grad = state.grad.detach().clone()
    expected_weight_grad = conv.weight.grad.detach().clone()
    expected_bias_grad = conv.bias.grad.detach().clone()

    conv.zero_grad(set_to_none=True)
    inputs.grad = None
    state.grad = None

    actual = conv(inputs, state).sum()
    actual.backward()

    assert torch.allclose(inputs.grad, expected_input_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(state.grad, expected_state_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(conv.weight.grad, expected_weight_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(conv.bias.grad, expected_bias_grad, atol=1e-6, rtol=1e-5)
