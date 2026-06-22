"""Component 3: fidelity_term (one LLM judge call per rollout).

A single extraction-style judge call covers both omission (recall of source
facts) and invention (unsupported claims). Scoring is done in code, not by
the judge:

    recall      = present_facts / total_facts        (1.0 if no facts)
    recall_term = max(0.2, recall)                    # linear, floored
    halluc_term = exp(-alpha * n_unsupported)         # smooth, uncapped
    fidelity_term = recall_term * halluc_term

The penalties are soft (x0.1 per claim, not a hard zero) so a single judge
misflag cannot zero a good rollout and dominate the group statistics. The
judge call fails OPEN: on parse/HTTP failure after one retry, the term is
1.0 and a judge_failure is counted — a missing verdict must not punish the
rollout.

Verdicts are cached in SQLite keyed on
sha256(model + prompt_version + source + candidate), so identical rollouts
within a group (common at low temperature) and re-scored eval outputs do
not pay twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from langsimp.verifier import BaseJudge, LocalJudge

_log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "reward.yaml"
_PROMPT_PATH = _ROOT / "prompts" / "fidelity_judge.txt"

_FLOOR = 0.2

# Hallucination penalty decay rate. halluc_term = exp(-alpha * n_unsupported),
# uncapped and smooth: n=1 -> ~0.30, n=2 -> ~0.09, n=3 -> ~0.027, n=5 -> ~0.002.
# Replaces the earlier capped `0.1 ** min(n, 2)`, which made every rollout with
# >=2 unsupported claims identical (0.01) — flattening the hardest groups to
# zero within-group variance (no GRPO gradient where it's most needed).
_HALLUC_ALPHA = 1.2

# OpenRouter / OpenAI structured-output schema for the verdict. `strict: true`
# forces the provider to emit JSON matching this exactly, eliminating the
# parse-failure tail seen on long free-form replies. Mirror this shape in
# prompts/fidelity_judge.txt and is_valid_verdict().
FIDELITY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "fidelity_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "source_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string"},
                            "tier": {"type": "string", "enum": ["core", "peripheral"]},
                            "status": {"type": "string", "enum": ["present", "absent"]},
                        },
                        "required": ["fact", "tier", "status"],
                        "additionalProperties": False,
                    },
                },
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["source_facts", "unsupported_claims"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class FidelityResult:
    term: float
    recall_term: float
    halluc_term: float
    recall: float
    n_unsupported: int
    judge_json: dict | None
    failed: bool
    from_cache: bool


# ---------- scoring (pure, no network) ----------


def is_valid_verdict(parsed: object) -> bool:
    """A usable judge verdict: a dict with list-typed source_facts and
    unsupported_claims. Anything else (incl. LocalJudge's {"error": ...})
    triggers the retry / fail-open path."""
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("source_facts"), list)
        and isinstance(parsed.get("unsupported_claims"), list)
    )


def score_verdict(parsed: dict) -> tuple[float, float, float, float, int]:
    """(term, recall_term, halluc_term, recall, n_unsupported) from a valid
    verdict. Assumes is_valid_verdict(parsed).

    v11.1: recall is computed over CORE facts only — dropping peripheral detail
    is expected in an A2 simplification and must not be penalized. A fact with a
    missing/unknown tier is treated as core (conservative: don't silently let an
    untiered fact escape the recall denominator). If there are no core facts,
    recall_term = 1.0 (a `no_core_facts` event for the caller to log).
    """
    facts = [f for f in parsed["source_facts"] if isinstance(f, dict)]
    core = [f for f in facts if f.get("tier", "core") != "peripheral"]
    if core:
        present = sum(1 for f in core if f.get("status") == "present")
        recall = present / len(core)
    else:
        recall = 1.0
    recall_term = max(_FLOOR, recall)

    n_unsupported = len(parsed["unsupported_claims"])
    halluc_term = math.exp(-_HALLUC_ALPHA * n_unsupported)

    return recall_term * halluc_term, recall_term, halluc_term, recall, n_unsupported


def has_core_facts(parsed: dict) -> bool:
    """False when the judge returned zero core facts (the `no_core_facts` case —
    rare, contentless sources). Used for logging/auditing only."""
    return any(
        isinstance(f, dict) and f.get("tier", "core") != "peripheral"
        for f in parsed.get("source_facts", [])
    )


def _result_from_verdict(parsed: dict, from_cache: bool) -> FidelityResult:
    term, recall_term, halluc_term, recall, n = score_verdict(parsed)
    return FidelityResult(
        term=term,
        recall_term=recall_term,
        halluc_term=halluc_term,
        recall=recall,
        n_unsupported=n,
        judge_json=parsed,
        failed=False,
        from_cache=from_cache,
    )


_FAIL_OPEN = FidelityResult(
    term=1.0, recall_term=1.0, halluc_term=1.0, recall=1.0,
    n_unsupported=0, judge_json=None, failed=True, from_cache=False,
)


# ---------- cache ----------


class JudgeCache:
    """Thread-safe SQLite verdict cache. Pass path=":memory:" for tests."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS verdicts (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM verdicts WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts (k, v) VALUES (?, ?)", (key, json.dumps(value))
            )
            self._conn.commit()


# ---------- scorer ----------


class FidelityScorer:
    """Per-rollout fidelity judge with caching, one retry, and fail-open."""

    def __init__(
        self,
        judge: BaseJudge,
        model_id: str,
        prompt_template: str,
        prompt_version: str,
        cache: JudgeCache | None = None,
    ):
        self.judge = judge
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.prompt_version = prompt_version
        self.cache = cache
        # Counters for the §5/§6 readouts (judge failure rate, cache hit rate).
        self.calls = 0
        self.failures = 0
        self.cache_hits = 0

    def _key(self, source: str, candidate: str) -> str:
        h = hashlib.sha256()
        for part in (self.model_id, self.prompt_version, source, candidate):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _render(self, source: str, candidate: str) -> str:
        # The template contains literal JSON braces, so substitute the two
        # placeholders by replacement rather than str.format().
        return self.prompt_template.replace("{source}", source).replace("{candidate}", candidate)

    def _call_with_retry(self, source: str, candidate: str) -> dict | None:
        prompt = self._render(source, candidate)
        for attempt in range(2):  # initial try + one retry
            self.calls += 1
            try:
                result = self.judge.evaluate(prompt)
            except Exception as e:  # pragma: no cover - transport already retries internally
                _log.warning("fidelity judge raised (attempt %d): %s", attempt, e)
                continue
            if is_valid_verdict(result):
                return result
            _log.warning("fidelity judge bad verdict (attempt %d): %.200r", attempt, result)
        return None

    def score(self, source: str, candidate: str) -> FidelityResult:
        key = self._key(source, candidate)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return _result_from_verdict(cached, from_cache=True)

        parsed = self._call_with_retry(source, candidate)
        if parsed is None:
            self.failures += 1
            _log.warning("judge_failure: fidelity failing open to 1.0")
            return _FAIL_OPEN

        if self.cache is not None:
            self.cache.put(key, parsed)
        return _result_from_verdict(parsed, from_cache=False)


# ---------- construction from config ----------


def load_fidelity_config(path: str | Path = _CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())["fidelity"]


def build_scorer(
    judge: BaseJudge | None = None,
    *,
    use_cache: bool = True,
    config: dict | None = None,
) -> FidelityScorer:
    """Construct a FidelityScorer from config/reward.yaml. If `judge` is None,
    build a LocalJudge against OpenRouter (needs OPENROUTER_API_KEY)."""
    cfg = config or load_fidelity_config()
    if judge is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (fidelity judge)")
        judge = LocalJudge(
            base_url=cfg["base_url"],
            model_name=cfg["model"],
            temperature=cfg.get("temperature", 0.0),
            api_key=api_key,
            max_tokens=cfg.get("max_tokens", 1500),
            response_format=FIDELITY_RESPONSE_FORMAT,
        )
    cache = JudgeCache(_ROOT / cfg["cache_path"]) if use_cache else None
    return FidelityScorer(
        judge=judge,
        model_id=cfg["model"],
        prompt_template=_PROMPT_PATH.read_text(),
        prompt_version=cfg["prompt_version"],
        cache=cache,
    )
