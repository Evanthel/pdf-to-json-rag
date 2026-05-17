"""Project paths and lightweight local configuration."""

from dataclasses import dataclass
import os
from pathlib import Path


DATA_DIR_ENV = "PDF_TO_JSON_RAG_DATA_DIR"
PROJECT_ROOT_ENV = "PDF_TO_JSON_RAG_PROJECT_ROOT"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    data_input: Path
    data_documents: Path
    data_chunks: Path
    data_index: Path
    data_eval: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        data_dir = root / "data"
        return cls(
            root=root,
            data_dir=data_dir,
            data_input=data_dir / "input",
            data_documents=data_dir / "documents",
            data_chunks=data_dir / "chunks",
            data_index=data_dir / "index",
            data_eval=data_dir / "eval",
        )

    @classmethod
    def from_data_dir(cls, root: Path, data_dir: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data_dir=data_dir,
            data_input=data_dir / "input",
            data_documents=data_dir / "documents",
            data_chunks=data_dir / "chunks",
            data_index=data_dir / "index",
            data_eval=data_dir / "eval",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.data_input,
            self.data_documents,
            self.data_chunks,
            self.data_index,
            self.data_eval,
        ):
            path.mkdir(parents=True, exist_ok=True)

def _resolve_project_root() -> Path:
    configured_root = os.getenv(PROJECT_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> ProjectPaths:
    project_root = _resolve_project_root()
    configured_data_dir = os.getenv(DATA_DIR_ENV)
    if configured_data_dir:
        data_dir = Path(configured_data_dir).expanduser().resolve()
        return ProjectPaths.from_data_dir(project_root, data_dir)
    return ProjectPaths.from_root(project_root)


PROJECT_ROOT = _resolve_project_root()
PATHS = _resolve_paths()
