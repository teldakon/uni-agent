"""Evaluate the prepared pytest task and require every expected test result."""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath

from unidiff import PatchSet

logger = logging.getLogger(__name__)

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
_STATUSES = "PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS"


def parse_pytest_output(output: str, metadata: dict) -> dict[str, str]:
    """Accept pytest's verbose lines and summary lines, retaining mixed results."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output).replace("\r", "")
    observed = []
    for line in clean.splitlines():
        line = line.strip()
        prefix = re.match(rf"^({_STATUSES})\s+(.+?)(?:\s+-\s+.*)?$", line)
        if prefix:
            observed.append((prefix.group(2), prefix.group(1).lower()))
            continue
        suffix = re.match(rf"^(.+?::.+?)\s+({_STATUSES})(?:\s+\[[^]]+\])?$", line)
        if suffix:
            observed.append((suffix.group(1), suffix.group(2).lower()))

    result = {}
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for node in metadata[category]:
            statuses = {
                status
                for payload, status in observed
                if payload == node or ("[" not in node and payload.startswith(node + "["))
            }
            if len(statuses) == 1:
                result[node] = next(iter(statuses))
            elif statuses:
                result[node] = "mixed:" + ",".join(sorted(statuses))
            else:
                result[node] = "missing"
    return result


def get_eval_report(metadata: dict, output: str, exit_code: int) -> dict:
    """A missing test or incomplete evaluation cannot resolve the task."""
    _, start, test_output = output.partition(START_TEST_OUTPUT)
    test_output, end, _ = test_output.partition(END_TEST_OUTPUT)
    report = {
        "resolved": False,
        "found_eval_status": bool(start and end),
        "test_status": None,
        "status_map": {},
    }
    if not report["found_eval_status"]:
        return report

    status_map = parse_pytest_output(test_output, metadata)
    test_status = {
        category: {
            "success": [node for node in metadata[category] if status_map[node] == "passed"],
            "failure": [node for node in metadata[category] if status_map[node] != "passed"],
        }
        for category in ("FAIL_TO_PASS", "PASS_TO_PASS")
    }
    report.update(status_map=status_map, test_status=test_status)
    report["resolved"] = (
        exit_code == 0
        and bool(metadata["FAIL_TO_PASS"])
        and not any(category["failure"] for category in test_status.values())
    )
    return report


def _test_paths(test_patch: str) -> list[str]:
    paths = set()
    for patched_file in PatchSet(test_patch):
        for raw in (patched_file.source_file, patched_file.target_file):
            if raw == "/dev/null":
                continue
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Test patch contains a path outside the repository")
            paths.add(str(path))
    return sorted(paths)


async def compute_reward(metadata: dict, sandbox, eval_timeout: float = 600.0) -> dict:
    """Restore official test files and run the patch and tests inside Nono."""
    if metadata["log_parser"] != "nono_pytest":
        raise ValueError("swe_nono_test requires the nono_pytest log parser")
    workdir = Path(sandbox.workdir)
    repo_dir = Path(sandbox.repo_dir)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="eval-", dir=sandbox.session_dir) as temporary:
        staging = Path(temporary)
        commands = ["set -eo pipefail"]
        for index, path in enumerate(_test_paths(metadata["test_patch"])):
            source = repo_dir / path
            target = workdir / path
            commands.append(f"rm -f -- {shlex.quote(str(target))}")
            if source.is_file():
                copied = staging / str(index)
                shutil.copy2(source, copied)
                commands.extend(
                    [
                        f"mkdir -p -- {shlex.quote(str(target.parent))}",
                        f"cp -p -- {shlex.quote(str(copied))} {shlex.quote(str(target))}",
                    ]
                )
        if metadata["install"]:
            commands.append(metadata["install"])
        test_patch = staging / "test.patch"
        test_patch.write_text(metadata["test_patch"], encoding="utf-8")
        commands.extend(
            [
                f"git apply -v -- {shlex.quote(str(test_patch))}",
                f"printf '%s\\n' '{START_TEST_OUTPUT}'",
                "set +e",
                metadata["test_cmd"],
                "test_status=$?",
                "set -e",
                f"printf '%s\\n' '{END_TEST_OUTPUT}'",
                'exit "$test_status"',
            ]
        )
        script = staging / "eval.sh"
        script.write_text("\n".join(["#!/bin/bash", *commands]) + "\n", encoding="utf-8")
        response = await sandbox.exec_shell(
            f"bash {shlex.quote(str(script))} 2>&1", workdir=str(workdir), timeout=eval_timeout
        )

    output = response.stdout
    if response.stderr:
        output += "\n" + response.stderr
    logger.info("nono eval output:\n%s", output)
    report = get_eval_report(metadata, output, response.exit_code)
    elapsed = time.perf_counter() - started
    logger.info("nono eval done: exit=%s resolved=%s", response.exit_code, report["resolved"])
    return {
        "eval_completed": report["found_eval_status"] and response.exit_code in (0, 1),
        "eval_execution_time": elapsed,
        "eval_exit_code": response.exit_code,
        "eval_output": output,
        "eval_report": report,
        "resolved": report["resolved"],
    }
