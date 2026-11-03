# Beaker: PR-Agent `/review` on Qodo PR-Review-Bench

Optimizes the key-issue findings of PR-Agent's `/review` tool against the planted issues in
[Qodo/PR-Review-Bench](https://huggingface.co/datasets/Qodo/PR-Review-Bench) (MIT). Only `/review`
is in scope. Beaker agent: **PR-Agent Review** (`pr-agent-review`).

## Editable surface

`repository(...)` is narrowed to the review path:

- `pr_agent/tools/pr_reviewer.py`
- `pr_agent/settings/pr_reviewer_prompts.toml`
- `pr_agent/algo/pr_processing.py`
- `pr_agent/algo/review_merge.py`

`configuration.toml` is excluded: Beaker targets are files, not TOML sections. The baseline keeps
the upstream `pr_reviewer.num_max_findings` cap; a candidate can still change it from
`pr_reviewer.py`.

## Data

`build_dataset.py` reads the benchmark at a pinned Hugging Face revision and snapshots each PR's
diff, title, body and head SHA through `gh api`, so cases never call GitHub. Generated JSONL only
exists in a temporary directory; the script smokes each dataset and uploads it:

```bash
uv run python .beaker/build_dataset.py              # smoke, then upload both datasets
uv run python .beaker/build_dataset.py --no-upload  # smoke only
```

- `qodo-pr-review-bench`: 100 PRs, 580 issues. Split **by repository** (see `splits.json`) so the
  test split measures generalization rather than repo-specific prompt tuning: test = prefect + tauri
  (26 PRs), train = the other six repositories (74 PRs).
- `qodo-pr-review-bench-quickstart`: 6 train / 3 test PRs for cheap end-to-end checks.

Issues with a `rule_name` are rule violations; the rest are functional bugs. 37 issues have no
file or line location.

**Recipe decision:** each repository's extracted rules are passed the way a real user would, as a
short list in `pr_reviewer.extra_instructions`. Without them, rule-violation recall is close to 0.

## Execution

`run_case` runs PR-Agent's real `/review` in plain-diff mode in-process, inside
`pr_agent.cli._cli_settings_scope()` (per-context settings, so cases can overlap). Review JSON goes
to a per-case temporary file. Nothing is published anywhere.

- Patch-only: the plain-diff provider's `.git` lookup is disabled for the case, so it never reads
  working-tree files from whatever checkout the evaluator runs in.
- Plain-diff mode gives the model no PR title or description (the provider hard-codes them).
  They are kept in the case input for the case viewer only.
- With no selected model, PR-Agent's upstream defaults (`config.model`, `fallback_models`) apply.
  With a selected comparison model, the run goes through Beaker's gateway with no fallbacks.
- Only the review is traced (LiteLLM integration); the judge is never traced.
- A failed or unparseable review raises; provider rate limits and timeouts are retryable.

Output: `findings`, one per `review.key_issues_to_review` entry, plus `security_concerns` when it
is not "No".

## Scoring

Beaker's built-in judge (`scorer_model: openai:gpt-6-luna` on the integration, through
`scoring_inference_target()`) makes one call per case. It matches each known issue to at most one
finding on substance, not wording. Invalid matchings are retried, then the case fails; they are
never scored as "not matched".

- Per case: recall = matched issues / issues, precision = matched findings / findings
  (0 findings → 0), F1 from those.
- Objective: F1. Field scores: `f1`, `recall`, `precision`, `recall_bugs`, `recall_rules` (the
  last two only on PRs that have issues of that kind).
- Checks: one per known issue (`bug` / `rule`), one failing check per unmatched finding
  (`false positive`), and an informational finding count.

The run objective is the **mean of per-case F1**, not Qodo's pooled F1, and our judge is not
Qodo's (their scoring code is not public). Numbers are not comparable to Qodo's published results.
