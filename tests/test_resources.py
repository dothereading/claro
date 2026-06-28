"""Tests for the shared repo-file loader (claro.resources).

The loader is the one place that knows where the repo root is and how to read
YAML / text. These pin its contract: repo-relative paths resolve against the
root, absolute paths pass through, and YAML parses to Python objects.
"""

from __future__ import annotations

import pytest

from claro import resources


class TestResolve:
    def test_repo_root_contains_marker_files(self):
        # If REPO_ROOT is wrong, everything else silently reads the wrong files.
        assert (resources.REPO_ROOT / "pyproject.toml").exists()
        assert (resources.REPO_ROOT / "CLAUDE.md").exists()

    def test_relative_path_resolves_against_repo_root(self):
        assert resources.resolve("config/reward.yaml") == resources.REPO_ROOT / "config" / "reward.yaml"

    def test_absolute_path_passes_through(self, tmp_path):
        p = tmp_path / "x.yaml"
        assert resources.resolve(p) == p


class TestLoadYaml:
    def test_loads_repo_relative_yaml(self):
        data = resources.load_yaml("config/reward.yaml")
        assert isinstance(data, dict)
        assert "fidelity" in data

    def test_loads_absolute_yaml(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("a: 1\nb: [x, y]\n")
        assert resources.load_yaml(p) == {"a": 1, "b": ["x", "y"]}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            resources.load_yaml("config/does_not_exist.yaml")


class TestLoadText:
    def test_returns_raw_text(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("hello\nworld\n")
        assert resources.load_text(p) == "hello\nworld\n"
