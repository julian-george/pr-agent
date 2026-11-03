"""Beaker integration: optimize PR-Agent's /review findings on Qodo PR-Review-Bench.
Scoring and data choices are explained in .beaker/README.md."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
    scoring_inference_target,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Same judge as integrations.pr_agent.scorer_model, used when no hosted scorer target exists.
LOCAL_JUDGE_MODEL = "gpt-6-luna"
JUDGE_ATTEMPTS = 3


class PullRequestInput(BaseModel):
    repo: str
    pr_number: int
    head_sha: str
    title: str
    description: str
    diff: str = Field(min_length=1)
    repo_rules: list[str]


class Issue(BaseModel):
    id: str
    title: str
    description: str
    # 37 of the 580 benchmark issues carry no file or line location.
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    problematic_code_snippet: str
    rule_name: str | None = None


class Expected(BaseModel):
    issues: list[Issue] = Field(min_length=1)


class Metadata(BaseModel):
    repo: str
    num_issues: int


class Row(BaseModel):
    id: str = Field(min_length=1)
    input: PullRequestInput
    expected: Expected
    metadata: Metadata


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input=row.input.model_dump(mode="json"),
            expected=row.expected.model_dump(mode="json"),
            metadata=row.metadata.model_dump(mode="json"),
        )


def render_repo_rules(rules: list[str]) -> str:
    lines = "\n".join(f"- {rule}" for rule in rules)
    return f"This repository enforces the following rules. Flag any change that violates one:\n{lines}"


def is_transient_provider_error(error: BaseException) -> bool:
    import litellm

    transient_types = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
                       litellm.InternalServerError, litellm.ServiceUnavailableError)
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, transient_types):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def findings_from_review(review: dict) -> list[dict]:
    findings = [
        {
            "id": f"f{index}",
            "relevant_file": issue.get("relevant_file", ""),
            "start_line": issue.get("start_line"),
            "end_line": issue.get("end_line"),
            "issue_header": issue.get("issue_header", ""),
            "issue_content": issue.get("issue_content", ""),
        }
        for index, issue in enumerate(review.get("key_issues_to_review") or [], start=1)
    ]
    security_concerns = str(review.get("security_concerns") or "").strip()
    if security_concerns and not security_concerns.lower().startswith("no"):
        findings.append({"id": "security", "relevant_file": "", "start_line": None, "end_line": None,
                         "issue_header": "Security concerns", "issue_content": security_concerns})
    return findings


def configure_review_settings(settings, pull_request: PullRequestInput, json_output_path: Path,
                              runtime: RolloutRuntime) -> None:
    # Mirrors pr_agent.cli.run() for --diff-file --json-output review.
    settings.set("config.git_provider", "plain-diff")
    settings.set("plain_diff.content", pull_request.diff)
    settings.set("plain_diff.json_output_path", str(json_output_path))
    settings.set("config.publish_output", True)
    settings.set("CONFIG.CLI_MODE", True)
    # Raise instead of swallowing tool errors, so a failed review is never scored as "found nothing".
    settings.set("config.propagate_tool_errors", True)
    settings.set("pr_reviewer.extra_instructions", render_repo_rules(pull_request.repo_rules))
    if runtime.model:
        target = inference_target(runtime)
        settings.set("config.model", f"openai/{target.model}")
        settings.set("config.fallback_models", [])
        settings.set("openai.key", target.api_key)
        settings.set("openai.api_base", target.base_url)


async def run_case(*, case_input, runtime: RolloutRuntime) -> CaseResult:
    from beaker.tracing.integrations.litellm import registered

    from pr_agent.agent.pr_agent import PRAgent
    from pr_agent.algo.run_details import command_failed, init_run_details
    from pr_agent.cli import _cli_settings_scope
    from pr_agent.config_loader import get_settings
    from pr_agent.git_providers import plain_diff_provider

    pull_request = PullRequestInput.model_validate(case_input)
    # Patch-only review: never read files from whatever checkout the evaluator runs in.
    plain_diff_provider._find_repository_root = lambda: None
    with tempfile.TemporaryDirectory(prefix="pr-agent-review-") as temp_dir, _cli_settings_scope():
        json_output_path = Path(temp_dir) / "review.json"
        configure_review_settings(get_settings(), pull_request, json_output_path, runtime)
        init_run_details()
        try:
            with runtime.trace.stage("pr_agent.review") as stage:
                async with registered(runtime.trace) as litellm_trace:
                    try:
                        handled = await PRAgent().handle_request("local_diff", ["review"])
                    finally:
                        await litellm_trace.flush()
                stage.output({"handled": bool(handled)})
        except Exception as error:
            if is_transient_provider_error(error):
                raise RetryableCaseError(f"Transient model provider error: {error}") from error
            raise
        if not handled or command_failed():
            raise RuntimeError("PR-Agent /review did not complete successfully")
        if not json_output_path.exists():
            raise RuntimeError("PR-Agent /review produced no parseable review")
        review = json.loads(json_output_path.read_text(encoding="utf-8"))["review"]
    return CaseResult(output={"findings": findings_from_review(review)}, output_kind="record")


def build_judge_prompt(issues: list[dict], findings: list[dict]) -> str:
    known_issues = [{key: issue[key] for key in ("id", "title", "description", "file_path", "start_line",
                                                 "end_line", "problematic_code_snippet")} for issue in issues]
    return (
        "You grade an automated code reviewer against known issues planted in a pull request.\n"
        "Match each known issue to the reviewer finding that identifies the same underlying problem, or null.\n"
        "Judge on substance, not wording; file and line locations may differ slightly. Each finding may match "
        "at most one issue.\n"
        'Reply with JSON only: {"matches": {"<issue id>": "<finding id or null>", ...}} with every issue id '
        "exactly once.\n\n"
        f"Known issues:\n{json.dumps(known_issues, indent=1)}\n\n"
        f"Reviewer findings:\n{json.dumps(findings, indent=1)}"
    )


def parse_judge_matches(reply: str, issue_ids: list[str], finding_ids: set[str]) -> dict[str, str | None]:
    matches = json.loads(reply)["matches"]
    if sorted(matches) != sorted(issue_ids):
        raise ValueError("judge did not answer every issue exactly once")
    matched = [finding for finding in matches.values() if finding is not None]
    if not set(matched) <= finding_ids:
        raise ValueError("judge referenced an unknown finding id")
    if len(matched) != len(set(matched)):
        raise ValueError("judge matched one finding to several issues")
    return matches


async def judge_matches(issues: list[dict], findings: list[dict]) -> dict[str, str | None]:
    target = scoring_inference_target()
    if target is None:
        client, model = AsyncOpenAI(), LOCAL_JUDGE_MODEL
    else:
        client, model = AsyncOpenAI(base_url=target.base_url, api_key=target.api_key), target.model
    prompt = build_judge_prompt(issues, findings)
    issue_ids = [issue["id"] for issue in issues]
    finding_ids = {finding["id"] for finding in findings}
    last_error: Exception | None = None
    async with client:
        for _ in range(JUDGE_ATTEMPTS):
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            try:
                return parse_judge_matches(response.choices[0].message.content or "", issue_ids, finding_ids)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                last_error = error
    raise RuntimeError(f"Judge returned no valid matching after {JUDGE_ATTEMPTS} attempts: {last_error}")


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    issues = case.expected["issues"]
    findings = result.output["findings"]
    matches = await judge_matches(issues, findings) if findings else {issue["id"]: None for issue in issues}
    findings_by_id = {finding["id"]: finding for finding in findings}

    def recall_of(subset: list[dict]) -> float:
        return sum(matches[issue["id"]] is not None for issue in subset) / len(subset)

    recall = recall_of(issues)
    bug_issues = [issue for issue in issues if not issue["rule_name"]]
    rule_issues = [issue for issue in issues if issue["rule_name"]]
    precision = sum(finding is not None for finding in matches.values()) / len(findings) if findings else 0.0
    f1 = f1_score(precision, recall)
    checks = [
        Check(
            name=issue["title"],
            description=f"{issue['file_path']}:{issue['start_line']}-{issue['end_line']}",
            verdict="pass" if matches[issue["id"]] else "fail",
            message=findings_by_id[matches[issue["id"]]]["issue_header"] if matches[issue["id"]] else "missed",
            group="rule" if issue["rule_name"] else "bug",
        )
        for issue in issues
    ]
    matched_finding_ids = set(matches.values())
    checks += [
        Check(
            name=finding["issue_header"] or finding["id"],
            description=f"{finding['relevant_file']}:{finding['start_line']}-{finding['end_line']}",
            verdict="fail",
            message="Finding matched no known issue",
            group="false positive",
        )
        for finding in findings
        if finding["id"] not in matched_finding_ids
    ]
    # Field scores must lie in [0, 1], so the raw finding count is reported as a check.
    checks.append(Check(name="Findings reported", verdict="pass", message=f"{len(findings)} findings",
                        informational=True))
    field_scores = {"f1": f1, "recall": recall, "precision": precision}
    # Recall over an empty subset is undefined; leave the field out rather than report 0.
    if bug_issues:
        field_scores["recall_bugs"] = recall_of(bug_issues)
    if rule_issues:
        field_scores["recall_rules"] = recall_of(rule_issues)
    return CaseScore(objective=f1, field_scores=field_scores, checks=tuple(checks))


integration = Integration(
    targets=repository((
        "pr_agent/tools/pr_reviewer.py",
        "pr_agent/settings/pr_reviewer_prompts.toml",
        "pr_agent/algo/pr_processing.py",
        "pr_agent/algo/review_merge.py",
    )),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
