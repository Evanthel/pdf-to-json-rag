"""Project paths and lightweight local configuration."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys


DATA_DIR_ENV = "PDF_TO_JSON_RAG_DATA_DIR"
PROJECT_ROOT_ENV = "PDF_TO_JSON_RAG_PROJECT_ROOT"
PROJECT_NAME = "pdf-to-json-rag"


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


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "pdf_to_json_rag").exists():
            return candidate
    return None


def _default_user_data_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / PROJECT_NAME
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / PROJECT_NAME
        return home / "AppData" / "Roaming" / PROJECT_NAME
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser().resolve() / PROJECT_NAME
    return home / ".local" / "share" / PROJECT_NAME


def _resolve_project_root() -> Path:
    configured_root = os.getenv(PROJECT_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    cwd_root = _find_project_root(Path.cwd())
    if cwd_root is not None:
        return cwd_root
    module_root = _find_project_root(Path(__file__).resolve())
    if module_root is not None:
        return module_root
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> ProjectPaths:
    project_root = _resolve_project_root()
    configured_data_dir = os.getenv(DATA_DIR_ENV)
    if configured_data_dir:
        data_dir = Path(configured_data_dir).expanduser().resolve()
        return ProjectPaths.from_data_dir(project_root, data_dir)
    if (project_root / "pyproject.toml").exists():
        return ProjectPaths.from_root(project_root)
    return ProjectPaths.from_data_dir(project_root, _default_user_data_dir())


PROJECT_ROOT = _resolve_project_root()
PATHS = _resolve_paths()
