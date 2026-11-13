"""Run the prepared SWE example in an isolated Nono session."""

from __future__ import annotations

import logging

from pydantic import Field, model_validator

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_reward

logger = logging.getLogger(__name__)


class SWENonoTestTaskConfig(TaskConfig):
    name: str = "swe_nono_test"
    run_oracle_solution: bool = False
    run_baseline: bool = False
    eval_timeout: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def _check_modes(self) -> SWENonoTestTaskConfig:
        if self.run_oracle_solution and self.run_baseline:
            raise ValueError("Choose either baseline or oracle mode")
        return self


@register_task("swe_nono_test")
class SWENonoTestTask(Task):
    config_model = SWENonoTestTaskConfig

    async def run(self) -> TaskResult:
        cfg: SWENonoTestTaskConfig = self.config  # type: ignore[assignment]
        metadata = cfg.metadata
        logger.info("starting swe_nono_test (instance_id=%s)", metadata["instance_id"])

        async with self.build_sandbox() as sandbox:
            workdir = str(sandbox.workdir)
            if metadata["install"]:
                installation = await sandbox.exec_shell(metadata["install"], workdir=workdir, timeout=cfg.eval_timeout)
                if installation.exit_code != 0:
                    raise RuntimeError(
                        "Failed to register the session's user-site installation:\n"
                        + installation.stdout
                        + installation.stderr
                    )

            agent_error = None
            if cfg.run_oracle_solution:
                patch_file = sandbox.session_dir / "gold.patch"
                await sandbox.write_file(str(patch_file), metadata["patch"])
                applied = await sandbox.exec(
                    ["git", "apply", "--whitespace=fix", str(patch_file)],
                    workdir=workdir,
                    timeout=cfg.eval_timeout,
                )
                if applied.exit_code != 0:
                    raise RuntimeError("Failed to apply gold patch: " + applied.stdout + applied.stderr)
                finished = True
            elif cfg.run_baseline:
                finished = True
            else:
                agent_result = await self.build_agent().run(sandbox=sandbox, messages=cfg.prompt, workdir=workdir)
                finished = agent_result.finished
                agent_error = agent_result.info.get("error")

            result = await compute_reward(metadata, sandbox, eval_timeout=cfg.eval_timeout)
            if isinstance(agent_error, str) and agent_error.strip():
                result["agent_error"] = agent_error
            logger.info("swe_nono_test done: resolved=%s", result["resolved"])
            return TaskResult(
                reward=float(result["resolved"]),
                accuracy=float(result["resolved"]),
                finished=finished,
                extra_info=result,
            )
