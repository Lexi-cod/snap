#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from server.runtime_config import runtime_readiness, validate_runtime_config


def main() -> int:
    validate_runtime_config()
    readiness = runtime_readiness()
    print(json.dumps({
        "ready": readiness["ready"],
        "required_missing_artifacts": readiness["required_missing_artifacts"],
        "task_count": len(readiness["tasks"]),
        "artifact_count": len(readiness["artifacts"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
