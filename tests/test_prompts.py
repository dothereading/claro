"""Tests for the YAML-backed prompt loader.

These don't pin specific text — that would just duplicate prompts.yaml. They
verify the loader contract: YAML keys are exposed as the right module-level
constants, all are non-empty strings, and the file is well-formed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import langsimp.prompts as prompts

EXPECTED_CONSTANTS = (
    "DISTILL_SYSTEM_PROMPT",
    "SFT_SYSTEM_PROMPT",
    "REJECTED_SUMMARIZE_PROMPT",
    "REJECTED_ELI5_PROMPT",
    "REJECTED_CLARIFY_PROMPT",
)


class TestPromptLoader:
    def test_all_constants_exposed(self):
        for name in EXPECTED_CONSTANTS:
            assert hasattr(prompts, name), f"missing constant: {name}"

    def test_all_constants_are_nonempty_strings(self):
        for name in EXPECTED_CONSTANTS:
            v = getattr(prompts, name)
            assert isinstance(v, str), f"{name} is {type(v).__name__}, not str"
            assert v.strip(), f"{name} is empty / whitespace-only"

    def test_distill_prompt_keeps_section_headers(self):
        # The distill prompt is the only multi-section one — verify the
        # markdown structure survives the YAML round-trip (block scalar
        # rather than folded scalar).
        text = prompts.DISTILL_SYSTEM_PROMPT
        assert "## Syntax (hard rules)" in text
        assert "## Length and content" in text
        assert "## Register and rhythm" in text
        assert "## Output" in text
        # Sections separated by blank lines, not collapsed into one paragraph.
        assert "\n\n## Syntax" in text

    def test_sft_prompt_is_single_line(self):
        # The SFT prompt is a folded scalar — newlines collapse to spaces.
        assert "\n" not in prompts.SFT_SYSTEM_PROMPT
        assert "CEFR A2" in prompts.SFT_SYSTEM_PROMPT

    def test_yaml_file_parses_independently(self):
        # Catches the case where prompts.yaml is malformed but somehow
        # importing langsimp.prompts didn't raise (e.g. cached import).
        path = Path(prompts.__file__).parent / "prompts.yaml"
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert set(data) >= {
            "distill_system",
            "sft_system",
            "rejected_summarize",
            "rejected_eli5",
            "rejected_clarify",
        }

    def test_missing_key_in_yaml_raises_keyerror(self, tmp_path):
        # If someone deletes a key from prompts.yaml, lookup on the parsed
        # dict should raise KeyError loudly — not return None and propagate
        # an empty prompt downstream.
        bad_yaml = tmp_path / "prompts.yaml"
        bad_yaml.write_text("distill_system: |-\n  hello\n")  # missing the rest
        data = yaml.safe_load(bad_yaml.read_text())
        with pytest.raises(KeyError):
            _ = data["sft_system"]
