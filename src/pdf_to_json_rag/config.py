"""Project paths and lightweight local configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths.from_root(PROJECT_ROOT)
