"""Tests for dataset record formatting and split logic.

Currently bridges between the OLD `prepare_mlx_data` module and the NEW
`datasets` module that the refactor will introduce. The shared behaviors:
  * `to_mlx_sft_record(complex, simple)` → MLX chat record
  * `to_mlx_dpo_record(prompt, chosen, rejected)` → MLX DPO record
  * `split_train_valid(rows, valid_frac, seed)` → (train, valid) lists
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _import_target():
    try:
        import mlx_data as ds  # type: ignore
        return ds, "new"
    except ImportError:
        import prepare_mlx_data as old  # type: ignore
        return old, "old"


mod, flavor = _import_target()


class TestSftRecord:
    def test_sft_record_shape(self):
        if flavor == "new":
            rec = mod.to_mlx_sft_record("complex text", "simple text")
        else:
            rec = mod.to_record("complex text", "simple text")
        assert "messages" in rec
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert rec["messages"][1]["content"] == "complex text"
        assert rec["messages"][2]["content"] == "simple text"

    def test_strips_whitespace_in_content(self):
        if flavor == "new":
            rec = mod.to_mlx_sft_record("  complex  ", "  simple  ")
        else:
            rec = mod.to_record("  complex  ", "  simple  ")
        assert rec["messages"][1]["content"] == "complex"
        assert rec["messages"][2]["content"] == "simple"


@pytest.mark.skipif(flavor == "old", reason="DPO record helper only exists in new datasets module")
class TestDpoRecord:
    def test_dpo_record_shape(self):
        rec = mod.to_mlx_dpo_record("prompt", "chosen", "rejected")
        assert set(rec.keys()) >= {"system", "prompt", "chosen", "rejected"}
        assert rec["prompt"] == "prompt"
        assert rec["chosen"] == "chosen"
        assert rec["rejected"] == "rejected"
        assert isinstance(rec["system"], str) and rec["system"]


@pytest.mark.skipif(flavor == "old", reason="split helper only exists in new datasets module")
class TestSplitTrainValid:
    def test_valid_size_respects_fraction(self):
        rows = [{"i": i} for i in range(100)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.1, seed=0)
        assert len(valid) == 10
        assert len(train) == 90

    def test_at_least_one_valid_row(self):
        rows = [{"i": i} for i in range(3)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.0, seed=0)
        assert len(valid) >= 1

    def test_seed_reproducible(self):
        rows = [{"i": i} for i in range(50)]
        a = mod.split_train_valid(rows, valid_frac=0.2, seed=42)
        b = mod.split_train_valid(rows, valid_frac=0.2, seed=42)
        assert a == b

    def test_train_and_valid_disjoint(self):
        rows = [{"i": i} for i in range(50)]
        train, valid = mod.split_train_valid(rows, valid_frac=0.2, seed=1)
        train_ids = {r["i"] for r in train}
        valid_ids = {r["i"] for r in valid}
        assert not (train_ids & valid_ids)
        assert train_ids | valid_ids == set(range(50))
