from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from server.runtime_config import REPO_ROOT, runtime_readiness


class RuntimeNotReady(RuntimeError):
    """Raised when a QNN-backed runtime method is called before assets exist."""


@dataclass(frozen=True)
class RuntimeReadiness:
    ready: bool
    missing_required_artifacts: tuple[str, ...]


class SnapOnRuntime:
    """Interface the Android runtime should mirror for local inference calls."""

    def readiness(self) -> RuntimeReadiness:
        raise NotImplementedError

    def answer_visual_question(
        self,
        question: str,
        image_bytes: bytes | None = None,
        memories: Iterable[str] = (),
    ) -> str:
        raise NotImplementedError

    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError

    def synthesize(self, text: str) -> bytes | None:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError


class ExecuTorchQnnRuntime(SnapOnRuntime):
    """
    Placeholder for the phone-native ExecuTorch/QNN implementation.

    The current repo has no Android project or checked-in .pte files yet. This
    class gives app code and tests a stable boundary while keeping runtime claims
    honest until the model artifacts and JNI/Kotlin bridge land.
    """

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root

    def readiness(self) -> RuntimeReadiness:
        status = runtime_readiness(self.repo_root)
        return RuntimeReadiness(
            ready=bool(status["ready"]),
            missing_required_artifacts=tuple(status["required_missing_artifacts"]),
        )

    def _require_ready(self, task: str) -> None:
        readiness = self.readiness()
        if not readiness.ready:
            missing = ", ".join(readiness.missing_required_artifacts)
            raise RuntimeNotReady(f"{task} runtime is missing required artifacts: {missing}")

    def answer_visual_question(
        self,
        question: str,
        image_bytes: bytes | None = None,
        memories: Iterable[str] = (),
    ) -> str:
        self._require_ready("visual_question_answering")
        raise NotImplementedError("ExecuTorch/QNN VLM invocation is not wired yet")

    def transcribe(self, audio_bytes: bytes) -> str:
        self._require_ready("speech_to_text")
        raise NotImplementedError("ExecuTorch/QNN Whisper invocation is not wired yet")

    def synthesize(self, text: str) -> bytes | None:
        return None

    def embed_text(self, text: str) -> list[float]:
        self._require_ready("memory_text_embedding")
        raise NotImplementedError("ExecuTorch/QNN embedding invocation is not wired yet")
