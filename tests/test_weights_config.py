from __future__ import annotations

import pytest

from scripts.tune_weights import generate_candidate_configs
from src.rank.weights import Stage3Weights


def test_stage3_weights_require_sum_of_one() -> None:
    with pytest.raises(ValueError):
        Stage3Weights(stage2=0.80, urgency=0.10, ev=0.05)


def test_generate_candidate_configs_has_stable_order() -> None:
    configs = generate_candidate_configs(max_configs=3)

    assert [config.config_id for config in configs] == [
        "s2_t0.70_a0.20_k0.10_e0.10__s3_s0.80_u0.15_v0.05__log",
        "s2_t0.70_a0.05_k0.05_e0.05__s3_s0.80_u0.10_v0.10__log",
        "s2_t0.70_a0.05_k0.05_e0.05__s3_s0.80_u0.15_v0.05__log",
    ]
