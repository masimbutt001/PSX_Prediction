"""Download the data saved by the cloud daily run (the `data-store` branch) to this computer."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_BRANCH = "data-store"


@dataclass
class SyncResult:
    """Outcome of a data sync."""

    ok: bool
    message: str


def _git(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=300, check=False
    )


def sync_data_store(
    data_dir: Path, repo_dir: Optional[Path] = None, branch: str = DATA_BRANCH
) -> SyncResult:
    """Bring `data_dir` up to date with the cloud run's saved data.

    - `data_dir` is already a clone of the data branch: fast-forward pull.
    - `data_dir` is missing or empty: clone the data branch from this repository's origin.
    - `data_dir` holds data made on this computer: leave it alone.
    """
    try:
        if (data_dir / ".git").exists():
            res = _git(["pull", "--ff-only", "--quiet"], cwd=data_dir)
            if res.returncode != 0:
                return SyncResult(False, f"Could not update data: {res.stderr.strip()}")
            return SyncResult(True, f"Data is up to date with the '{branch}' branch.")

        if data_dir.exists() and any(data_dir.iterdir()):
            return SyncResult(
                False,
                f"'{data_dir}' already holds data made on this computer, so the cloud data "
                f"was not downloaded. Rename that folder and run 'psx data sync' to switch.",
            )

        origin = _git(["remote", "get-url", "origin"], cwd=repo_dir or Path.cwd())
        if origin.returncode != 0 or not origin.stdout.strip():
            return SyncResult(False, "Could not find this repository's GitHub address.")
        url = origin.stdout.strip()

        if _git(["ls-remote", "--exit-code", "--heads", url, branch]).returncode != 0:
            return SyncResult(
                False, f"No '{branch}' branch on GitHub yet: the cloud run has not saved data."
            )
        if data_dir.exists():
            data_dir.rmdir()  # empty, checked above
        res = _git(["clone", "--quiet", "--depth", "1", "--branch", branch, url, str(data_dir)])
        if res.returncode != 0:
            return SyncResult(False, f"Could not download data: {res.stderr.strip()}")
        return SyncResult(True, f"Downloaded the cloud run's data into '{data_dir}'.")
    except FileNotFoundError:
        return SyncResult(False, "Git is not installed, so data could not be synced.")
    except subprocess.TimeoutExpired:
        return SyncResult(False, "Timed out while syncing data from GitHub.")
