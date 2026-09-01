import torch

from bernini_v2.planner import maskgit_order


def test_maskgit_order_is_seeded_permutation_and_scatter_compatible():
    order = maskgit_order(8, 42)
    assert torch.equal(order, maskgit_order(8, 42))
    assert not torch.equal(order, maskgit_order(8, 43))
    assert order.dtype == torch.int64
    assert sorted(order.tolist()) == list(range(8))
    target = torch.zeros(8, dtype=torch.bool)
    target.scatter_(0, order[:3], True)
    assert target.sum() == 3


def test_maskgit_order_accepts_comfy_64_bit_seed_range():
    assert torch.equal(
        maskgit_order(8, 0xFFFFFFFFFFFFFFFF),
        maskgit_order(8, 0xFFFFFFFF),
    )
