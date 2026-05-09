"""Tests for the training-log parsers used to forward metrics to W&B.

The parsers are pure regex over real log lines; we test them against the
exact formats emitted by mlx-lm and mlx-lm-lora (sampled from logs/).
"""
from __future__ import annotations

import pytest

train = pytest.importorskip("train", reason="train.py not implemented yet")


class TestParseSftLine:
    def test_train_line(self):
        line = "Iter 10: Train loss 2.249, Learning Rate 1.000e-04, It/sec 1.380, Tokens/sec 404.447, Trained Tokens 2931, Peak mem 4.005 GB"
        m = train.parse_sft_line(line)
        assert m == {
            "iter": 10,
            "train/loss": 2.249,
            "train/lr": 1.000e-04,
            "train/it_per_sec": 1.380,
            "train/tok_per_sec": 404.447,
            "train/trained_tokens": 2931,
            "train/peak_mem_gb": 4.005,
        }

    def test_val_line(self):
        line = "Iter 50: Val loss 1.728, Val took 0.282s"
        m = train.parse_sft_line(line)
        assert m == {"iter": 50, "valid/loss": 1.728}

    def test_unrelated_line_returns_none(self):
        assert train.parse_sft_line("Loading model...") is None
        assert train.parse_sft_line("") is None

    def test_iter_1_val_works(self):
        # The very first val happens at iter 1 in the SFT log
        line = "Iter 1: Val loss 3.903, Val took 2.386s"
        assert train.parse_sft_line(line) == {"iter": 1, "valid/loss": 3.903}


class TestParseDpoLine:
    def test_train_line(self):
        line = "Iter 10: loss 0.011, chosen_r 83.361, rejected_r 65.698, acc 1.000, margin 17.663, lr 5.000e-06, it/s 2.012, tok/s 1172.116, peak_mem 8.719GB"
        m = train.parse_dpo_line(line)
        assert m == {
            "iter": 10,
            "train/loss": 0.011,
            "train/chosen_reward": 83.361,
            "train/rejected_reward": 65.698,
            "train/accuracy": 1.000,
            "train/margin": 17.663,
            "train/lr": 5.000e-06,
            "train/it_per_sec": 2.012,
            "train/tok_per_sec": 1172.116,
            "train/peak_mem_gb": 8.719,
        }

    def test_val_line(self):
        line = "Iter 50: Val loss 0.000, Val chosen reward 0.143, Val rejected reward 0.122, Val accuracy 1.000, Val margin 10.016, Val took 0.765s"
        m = train.parse_dpo_line(line)
        assert m == {
            "iter": 50,
            "valid/loss": 0.000,
            "valid/chosen_reward": 0.143,
            "valid/rejected_reward": 0.122,
            "valid/accuracy": 1.000,
            "valid/margin": 10.016,
        }

    def test_handles_negative_margin(self):
        line = "Iter 5: loss 0.500, chosen_r 10.000, rejected_r 12.000, acc 0.500, margin -2.000, lr 5.000e-06, it/s 2.000, tok/s 1000.000, peak_mem 8.000GB"
        m = train.parse_dpo_line(line)
        assert m["train/margin"] == -2.000

    def test_unrelated_line_returns_none(self):
        assert train.parse_dpo_line("De-quantizing model") is None
        assert train.parse_dpo_line("") is None


class TestBuildRunName:
    def test_includes_stage_and_model(self):
        name = train.build_run_name(stage="sft", model="mlx-community/gemma-3-1b-it-bf16",
                                    config={"iters": 300, "lr": 1e-4, "batch_size": 1})
        assert "sft" in name
        assert "gemma-3-1b" in name

    def test_includes_iters_and_lr(self):
        name = train.build_run_name(stage="dpo", model="mlx-community/gemma-3-1b-it-bf16",
                                    config={"iters": 500, "lr": 5e-6, "beta": 0.1})
        assert "iters500" in name
        assert "lr5e-06" in name or "lr5.0e-06" in name or "5e-06" in name
