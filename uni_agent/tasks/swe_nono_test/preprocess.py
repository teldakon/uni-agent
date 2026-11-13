"""Package one prepared local task for the nono sandbox.

The dataset root contains ``environment-tasks.yml`` and ``tasks/<instance_id>/``.
Each task directory must contain ``repo/``, ``task.yaml``, ``gold.patch``,
``test.patch``, ``problem_statement.md``, ``tests.json``, and
``install_config.yaml``. The referenced Conda environment must already exist.
Preprocessing reads these inputs without installing or executing task commands.

Example::

    LOCAL_DATASET_ROOT=/path/to/task python -m uni_agent.tasks.swe_nono_test.preprocess
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

LOCAL_DATASET_ROOT = os.environ.get("LOCAL_DATASET_ROOT") or None
TEMP_DIRECTORY_INSTRUCTIONS = (
    "Create test scripts, test outputs, and other temporary files in the current "
    "repository working directory or under $TMPDIR. Use relative paths or Python's "
    "tempfile module, which honors $TMPDIR. Do not hard-code /tmp or other paths "
    "outside these writable directories."
)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def build_swe_nono_test(local_dataset_root: str | Path | None = None) -> Dataset:
    """Read the single prepared task referenced by the local manifest."""
    if local_dataset_root is None:
        local_dataset_root = os.environ.get("LOCAL_DATASET_ROOT")
    if not local_dataset_root:
        raise ValueError("Set LOCAL_DATASET_ROOT or pass a local dataset root explicitly")
    root = Path(local_dataset_root).expanduser().resolve()
    manifest = _read_yaml_mapping(root / "environment-tasks.yml")
    task_index = manifest.get("task_index")
    if not isinstance(task_index, dict) or len(task_index) != 1:
        raise ValueError("swe_nono_test requires exactly one entry in environment-tasks.yml task_index")

    instance_id, task_entry = next(iter(task_index.items()))
    if not isinstance(instance_id, str) or Path(instance_id).name != instance_id or instance_id in {"", ".", ".."}:
        raise ValueError("The task instance_id must be a single directory name")
    if not isinstance(task_entry, dict):
        raise ValueError(f"Expected a manifest mapping for {instance_id}")
    environment_name = task_entry["environment"]
    environment = manifest["environments"][environment_name]
    conda_dir = (root / Path(environment["conda_dir"]).expanduser()).resolve()
    startup_commands = task_entry["startup_commands"]
    if not isinstance(startup_commands, list) or not all(isinstance(command, str) for command in startup_commands):
        raise ValueError(f"startup_commands for {instance_id} must be a list of strings")

    task_dir = root / "tasks" / instance_id
    repo_dir = (task_dir / "repo").resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"Prepared repository is missing: {repo_dir}")
    if not (conda_dir / "bin" / "python").is_file():
        raise FileNotFoundError(f"Prepared Conda Python is missing: {conda_dir / 'bin' / 'python'}")

    source_metadata = _read_yaml_mapping(task_dir / "task.yaml")
    if source_metadata.get("instance_id") != instance_id:
        raise ValueError(f"task.yaml instance_id does not match manifest entry {instance_id}")
    problem_statement = (task_dir / "problem_statement.md").read_text(encoding="utf-8")
    patch = (task_dir / "gold.patch").read_text(encoding="utf-8")
    test_patch = (task_dir / "test.patch").read_text(encoding="utf-8")
    tests = json.loads((task_dir / "tests.json").read_text(encoding="utf-8"))
    install_config = _read_yaml_mapping(task_dir / "install_config.yaml")

    metadata = {
        "instance_id": instance_id,
        "repo": source_metadata["repo"],
        "base_commit": source_metadata["base_commit"],
        "patch": patch,
        "test_patch": test_patch,
        "problem_statement": problem_statement,
        "environment_instructions": TEMP_DIRECTORY_INSTRUCTIONS,
        "FAIL_TO_PASS": tests["FAIL_TO_PASS"],
        "FAIL_TO_FAIL": tests.get("FAIL_TO_FAIL", []),
        "PASS_TO_PASS": tests["PASS_TO_PASS"],
        "PASS_TO_FAIL": tests.get("PASS_TO_FAIL", []),
        "install": install_config["install"],
        "log_parser": install_config["log_parser"],
        "test_cmd": install_config["test_cmd"],
    }
    task_config = {
        "name": "swe_nono_test",
        "sandbox": {
            "provider": "nono",
            "sandbox_kwargs": {
                "instance_id": instance_id,
                "repo_dir": str(repo_dir),
                "conda_dir": str(conda_dir),
                "startup_commands": startup_commands,
            },
        },
        "metadata": metadata,
    }
    row = {
        "data_source": "swe_nono_test",
        "prompt": [{"role": "user", "content": f"{problem_statement}\n\n{TEMP_DIRECTORY_INSTRUCTIONS}"}],
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }
    return Dataset.from_list([row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dataset-root",
        type=Path,
        default=LOCAL_DATASET_ROOT,
        help="Prepared dataset root (default: LOCAL_DATASET_ROOT environment variable).",
    )
    args = parser.parse_args()
    if args.local_dataset_root is None:
        parser.error("Set LOCAL_DATASET_ROOT or pass --local-dataset-root")
    dataset = build_swe_nono_test(args.local_dataset_root)
    out_path = args.local_dataset_root.expanduser().resolve() / "swe_nono_test.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instance to {out_path}", flush=True)


if __name__ == "__main__":
    main()
