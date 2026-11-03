"""Build, smoke and upload the Qodo PR-Review-Bench datasets (see .beaker/README.md).
Rows snapshot each PR's diff, title, body and head SHA, so cases never call GitHub."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BEAKER_DIR = Path(__file__).resolve().parent
HF_DATASET = "Qodo/PR-Review-Bench"
# Pin the Hugging Face revision so rebuilding produces the same labels.
HF_REVISION = "a73957c450a70693a743260e5637fffc44625f16"
BENCHMARK_FILE = "git_code_review_bench_100_w_open_prs.jsonl"
RULES_FILE = "rules_for_repo.jsonl"
GITHUB_OWNER = "agentic-review-benchmarks"
FULL_DATASET_NAME = "qodo-pr-review-bench"
QUICKSTART_DATASET_NAME = "qodo-pr-review-bench-quickstart"


def read_hugging_face_jsonl(filename: str) -> list[dict]:
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{filename}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]


def github_api(path: str, *, accept: str | None = None) -> str:
    command = ["gh", "api", path]
    if accept:
        command += ["-H", f"Accept: {accept}"]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def render_rule(rule: dict) -> str:
    return f"{rule['title']}: {rule['success_criteria']}"


def build_row(benchmark_row: dict, rules_by_repo: dict[str, list[str]]) -> dict:
    repo = benchmark_row["repo"]
    pr_number = int(benchmark_row["pr_url_to_review"].rstrip("/").split("/")[-1])
    row_id = f"{repo}-{pr_number}"
    api_path = f"repos/{GITHUB_OWNER}/{repo}/pulls/{pr_number}"
    pull_request = json.loads(github_api(api_path))
    diff = github_api(api_path, accept="application/vnd.github.diff")
    if not diff.strip():
        raise ValueError(f"{row_id}: GitHub returned an empty diff")
    issues = [
        {
            "id": f"{row_id}-{index}",
            "title": issue["title"],
            "description": issue["description"],
            "file_path": issue["file_path"],
            "start_line": issue["start_line"],
            "end_line": issue["end_line"],
            "problematic_code_snippet": issue["problematic_code_snippet"],
            "rule_name": issue.get("rule_name") or None,
        }
        for index, issue in enumerate(benchmark_row["issues"])
    ]
    if len(issues) != benchmark_row["num_of_issues"]:
        raise ValueError(f"{row_id}: {len(issues)} issues but num_of_issues is {benchmark_row['num_of_issues']}")
    return {
        "id": row_id,
        "input": {
            "repo": repo,
            "pr_number": pr_number,
            "head_sha": pull_request["head"]["sha"],
            "title": pull_request["title"],
            "description": pull_request["body"] or "",
            "diff": diff,
            "repo_rules": rules_by_repo[repo],
        },
        "expected": {"issues": issues},
        "metadata": {"repo": repo, "num_issues": len(issues)},
    }


def write_dataset(dataset_dir: Path, rows_by_id: dict[str, dict], split_ids: dict[str, list[str]]) -> None:
    dataset_dir.mkdir()
    for split_name, ids in split_ids.items():
        with (dataset_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
            for row_id in ids:
                output.write(json.dumps(rows_by_id[row_id]) + "\n")
    manifest = {"source": f"huggingface.co/datasets/{HF_DATASET}@{HF_REVISION}", "license": "MIT"}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def smoke(dataset_dir: Path) -> None:
    config = json.dumps({"local_dataset_path": str(dataset_dir)})
    subprocess.run(["beaker", "run", "smoke", "--strict", "--config", config], check=True)


def upload(dataset_dir: Path, name: str, split_ids: dict[str, list[str]]) -> str:
    command = ["beaker", "dataset", "upload", str(dataset_dir), "--name", name,
               "--total-count", str(sum(len(ids) for ids in split_ids.values()))]
    for split_name, ids in split_ids.items():
        command += ["--split", f"{split_name}={len(ids)}"]
    command.append("--json")
    artifact = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return f"{artifact['artifact_key']}@{artifact['dataset_revision']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-upload", action="store_true", help="Build and smoke locally without uploading.")
    args = parser.parse_args()

    splits = json.loads((BEAKER_DIR / "splits.json").read_text(encoding="utf-8"))
    rules_by_repo = {
        entry["repo"]: [render_rule(rule) for rule in entry["extracted_rules"]]
        for entry in read_hugging_face_jsonl(RULES_FILE)
    }
    benchmark_rows = read_hugging_face_jsonl(BENCHMARK_FILE)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda row: build_row(row, rules_by_repo), benchmark_rows))
    rows_by_id = {row["id"]: row for row in rows}
    if sorted(rows_by_id) != sorted(splits["train"] + splits["test"]):
        raise ValueError("splits.json does not cover exactly the benchmark's PRs")

    datasets = {
        FULL_DATASET_NAME: {"train": splits["train"], "test": splits["test"]},
        QUICKSTART_DATASET_NAME: splits["quickstart"],
    }
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        for name, split_ids in datasets.items():
            dataset_dir = Path(temp_dir) / name
            write_dataset(dataset_dir, rows_by_id, split_ids)
            smoke(dataset_dir)
            if not args.no_upload:
                print(f"{name}: {upload(dataset_dir, name, split_ids)}")


if __name__ == "__main__":
    main()
