# Project: Language Simplification LLM

Fine-tuning a small Gemma model to rewrite complex English at CEFR A2 level. See `docs/PLAN.md` for the staged pipeline (data → SFT → GSPO → eval). Abandoned arms (DPO, the v6–v10 reward iterations) live under `experiments/`.

## Code style: deep modules, not shallow scripts

Prefer a small number of well-organized files with classes and clear interfaces over many short single-purpose scripts. A "deep" module hides nontrivial implementation behind a small surface; a "shallow" module is mostly a thin pass-through.

Concretely:
- Group related logic into a single module with classes/functions, not many sibling files that each do one tiny thing.
- When two scripts share >30% of their logic, extract the common piece into one module and have both call into it. Don't duplicate.
- A new file needs a justification — a clearly distinct concern, or a clearly distinct consumer. "It's a different CLI" is not enough; CLI subcommands or thin wrappers are fine.
- Classes earn their keep when they hold state or polymorphism (e.g. `BaseJudge` / `LocalJudge`). Don't wrap stateless functions in classes for ceremony.

When refactoring an ad-hoc layout into a deeper one, write characterization tests for the behaviors you care about *first* (red/green still applies — see below), then move code, then confirm tests still pass.

## Development workflow: red/green TDD

For any new function, behavior change, or bug fix, follow test-first development:

1. **Red** — write a failing test first. Run it. Confirm it fails for the expected reason (not an import error or typo).
2. **Green** — write the minimum code to make the test pass. Run it. Confirm it passes.
3. **Refactor** — clean up if needed; tests must stay green.

Do not skip the red step. A test that passes before the implementation exists is not exercising the new code, and gives false confidence.

Keep tests close to the code they exercise. Prefer small, focused tests over large end-to-end ones for new logic. Use `pytest`.

### Exceptions (ask before skipping tests)

- Exploratory spikes you intend to throw away
- One-off data inspection or debugging scripts
- Prompt iteration where the "correctness" signal is human judgment
- Shell wrappers around existing tools (e.g. `train_*.sh`)

If unsure whether something needs tests, ask.

## Smoke-test before long runs

For any job expected to run more than ~10 minutes, do a 1–2 iteration / few-record version first and look at the outputs. Confirm the basics — values are real, generated text looks right — before committing to the full run.
