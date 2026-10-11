import torch

from experiments.range_search_w4.core import (
    candidate_grids, gptq_candidates, gptq_row, group_errors,
    make_grid, output_mse, prepare_hinv, quantize,
)


def test_grid_and_degenerate_rows():
    w = torch.tensor([-2.0, 0.5, 1.0])
    sym, count = candidate_grids(w, sym=True)
    asym, count2 = candidate_grids(w, sym=False)
    assert count == 10 and count2 == 100
    assert any(g.alpha == g.beta == 1 for g in sym)
    assert any(g.alpha == g.beta == 1 for g in asym)
    assert make_grid(torch.zeros(8)).scale == 2 / 15
    assert torch.isfinite(quantize(torch.zeros(8), make_grid(torch.zeros(8)))).all()
    assert torch.isfinite(quantize(torch.ones(8), make_grid(torch.ones(8), sym=False))).all()
    zeros, nominal = candidate_grids(torch.zeros(8), sym=False)
    assert nominal == 100 and len(zeros) == 1


def test_affine_grid_matches_gptqmodel_quantizer():
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.quantizer import Quantizer

    for sym, w in ((True, torch.tensor([0.5, 1.0, 2.0])),
                   (False, torch.tensor([-2.0, -0.3, 1.0]))):
        q = Quantizer(QuantizeConfig(bits=4, group_size=128, sym=sym, mse=0), name="local_check")
        q.configure(perchannel=True)
        q.find_params(w[None], weight=True)
        g = make_grid(w, sym=sym)
        assert abs(float(q.scale) - g.scale) < 1e-6
        assert float(q.zero) == g.zero
        assert torch.allclose(q.quantize(w[None]).flatten(), quantize(w, g))


def test_output_and_candidate_state():
    torch.manual_seed(7)
    x = torch.randn(16, 8)
    w = torch.randn(8)
    h = prepare_hinv(x, document_count=2)
    grids = [make_grid(w, a, a) for a in (1.0, 0.6, 0.3)]
    before = w.clone()
    batch = gptq_candidates(w, [[g] for g in grids], h, block_size=4, chunk_size=2)
    assert torch.equal(w, before)
    for i, g in enumerate(grids):
        assert torch.allclose(batch[i], gptq_row(w, [g], h, block_size=4), atol=1e-6)
        direct = ((x @ batch[i]) - (x @ w)).square().mean().item()
        assert abs(direct - output_mse(x, w, batch[i])) < 1e-6
    assert torch.allclose(gptq_candidates(w, [[grids[0]]], h)[0], batch[0])


def test_group_residual_identity():
    torch.manual_seed(11)
    x, w = torch.randn(13, 8), torch.randn(8)
    q = quantize(w, make_grid(w))
    d = group_errors(x, w, q, 4)
    assert abs(d["full_mse"] - output_mse(x, w, q)) < 1e-6
    assert abs(d["cross_term"] - (d["full_sum_squares"] - d["independent_sum_squares"])) < 1e-6


def test_minmax_matches_gptqmodel_per_channel():
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(17)
    x = torch.randn(2, 9, 8)
    layer = torch.nn.Linear(8, 3, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=-1, sym=True, desc_act=False, damp_percent=0.1, mse=0)
    reference = GPTQ(layer, qcfg=qcfg)
    # Direct GPTQ(module) uses the HF-Optimum compatibility name. Its
    # configure() defaults sym=False unless explicitly supplied.
    reference.quantizer.configure(perchannel=True, bits=4, sym=True)
    for doc in x:
        reference.process_batch(doc[None])
    actual, scales, zeros, *_ = reference.quantize(blocksize=4)
    hinv = prepare_hinv(x.reshape(-1, 8), document_count=2)
    expected = torch.stack([gptq_row(w, [make_grid(w)], hinv, block_size=4) for w in layer.weight])
    assert torch.allclose(actual.float(), expected, rtol=1e-4, atol=1e-4)
    assert torch.allclose(scales.flatten(), torch.tensor([make_grid(w).scale for w in layer.weight]), rtol=1e-5, atol=1e-6)
    assert torch.equal(zeros.flatten(), torch.full((3,), 8.0))


def test_external_group_grids_reach_gptqmodel():
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(19)
    x = torch.randn(2, 9, 8)
    layer = torch.nn.Linear(8, 1, bias=False)
    w = layer.weight.detach().float().flatten()
    grids = [make_grid(w[:4], 0.6, 0.6), make_grid(w[4:], 0.9, 0.9)]
    qcfg = QuantizeConfig(bits=4, group_size=4, sym=True, desc_act=False, damp_percent=0.1, mse=0)
    reference = GPTQ(layer, qcfg=qcfg)
    reference.quantizer.configure(perchannel=True, bits=4, sym=True)
    calls = []

    def fixed_find_params(_x, weight=False):
        assert weight
        group = min(max(len(calls) - 1, 0), 1)  # initial full-W call, then each group
        grid = grids[group]
        reference.quantizer.scale = torch.tensor([[grid.scale]])
        reference.quantizer.zero = torch.tensor([[grid.zero]])
        calls.append(group)

    reference.quantizer.find_params = fixed_find_params
    for doc in x:
        reference.process_batch(doc[None])
    actual, scales, zeros, *_ = reference.quantize(blocksize=4)
    h = prepare_hinv(x.reshape(-1, 8), document_count=2)
    expected = gptq_row(w, grids, h, group_size=4, block_size=4)
    assert calls == [0, 0, 1]
    assert torch.allclose(actual.float().flatten(), expected, atol=1e-5)
    assert torch.allclose(scales.flatten(), torch.tensor([g.scale for g in grids]))
    assert torch.allclose(zeros.flatten(), torch.tensor([g.zero for g in grids]))
