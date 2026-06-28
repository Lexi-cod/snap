import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TASKS_PATH = REPO_ROOT / "config" / "runtime_tasks.json"
ARTIFACT_MANIFEST_PATH = REPO_ROOT / "models" / "artifact_manifest.json"


class RuntimeConfigError(ValueError):
    """Raised when runtime configuration files are internally inconsistent."""


@dataclass(frozen=True)
class ArtifactStatus:
    id: str
    path: str
    required: bool
    present: bool
    sha256: str | None
    sha256_ok: bool | None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime_tasks(path: Path = RUNTIME_TASKS_PATH) -> dict[str, Any]:
    return _read_json(path)


def load_artifact_manifest(path: Path = ARTIFACT_MANIFEST_PATH) -> dict[str, Any]:
    return _read_json(path)


def validate_runtime_config(
    tasks: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    tasks = tasks or load_runtime_tasks()
    manifest = manifest or load_artifact_manifest()

    artifact_ids = {
        artifact["id"]
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("id")
    }
    if not artifact_ids:
        raise RuntimeConfigError("artifact manifest must define at least one artifact")

    task_ids: set[str] = set()
    for task in tasks.get("tasks", []):
        task_id = task.get("id")
        if not task_id:
            raise RuntimeConfigError("every runtime task must have an id")
        if task_id in task_ids:
            raise RuntimeConfigError(f"duplicate runtime task id: {task_id}")
        task_ids.add(task_id)

        for artifact_id in task.get("model_artifacts", []):
            if artifact_id not in artifact_ids:
                raise RuntimeConfigError(
                    f"task {task_id} references unknown artifact {artifact_id}"
                )

    for artifact in manifest.get("artifacts", []):
        if not artifact.get("path"):
            raise RuntimeConfigError(f"artifact {artifact.get('id')} is missing path")
        if os.path.isabs(artifact["path"]):
            raise RuntimeConfigError(
                f"artifact {artifact['id']} path must be repo-relative"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_statuses(repo_root: Path = REPO_ROOT) -> list[ArtifactStatus]:
    manifest = load_artifact_manifest()
    validate_runtime_config(manifest=manifest)

    statuses: list[ArtifactStatus] = []
    for artifact in manifest["artifacts"]:
        path = repo_root / artifact["path"]
        expected_sha = artifact.get("sha256")
        present = path.exists()
        sha_ok: bool | None = None
        actual_sha: str | None = None

        if present and expected_sha and expected_sha != "TODO" and path.is_file():
            actual_sha = _sha256(path)
            sha_ok = actual_sha == expected_sha
        elif present and expected_sha == "TODO":
            sha_ok = None

        statuses.append(
            ArtifactStatus(
                id=artifact["id"],
                path=artifact["path"],
                required=bool(artifact.get("required")),
                present=present,
                sha256=actual_sha,
                sha256_ok=sha_ok,
            )
        )
    return statuses


def runtime_readiness(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    tasks = load_runtime_tasks()
    manifest = load_artifact_manifest()
    validate_runtime_config(tasks=tasks, manifest=manifest)

    statuses = artifact_statuses(repo_root)
    status_by_id = {status.id: status for status in statuses}

    task_results = []
    for task in tasks["tasks"]:
        required_artifacts = [
            status_by_id[artifact_id]
            for artifact_id in task.get("model_artifacts", [])
            if status_by_id[artifact_id].required
        ]
        ready = all(status.present for status in required_artifacts)
        task_results.append(
            {
                "id": task["id"],
                "primary_runtime": task["primary_runtime"],
                "fallback_runtime": task["fallback_runtime"],
                "ready": ready,
                "missing_required_artifacts": [
                    status.id for status in required_artifacts if not status.present
                ],
            }
        )

    required_missing = [
        status.id for status in statuses if status.required and not status.present
    ]
    return {
        "target_device": tasks["target_device"],
        "ready": not required_missing,
        "required_missing_artifacts": required_missing,
        "tasks": task_results,
        "artifacts": [status.__dict__ for status in statuses],
    }


if __name__ == "__main__":
    print(json.dumps(runtime_readiness(), indent=2))
