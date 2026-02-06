from loguru import logger as loguru_logger
import subprocess
from typing import Dict, Any


logger = loguru_logger.bind(name="logging_utils")


def get_git_info() -> Dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        describe = subprocess.run(
            ["git", "-C", root, "describe", "--always", "--dirty", "--tags", "--long"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        dirty = (
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            != ""
        )

        return {
            "root": root,
            "commit": commit,
            "describe": describe,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return {
            "root": None,
            "commit": "unknown",
            "describe": "unknown",
            "branch": "unknown",
            "dirty": None,
        }
