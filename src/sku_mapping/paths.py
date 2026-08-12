"""Repository-root discovery and portable runtime-path helpers."""

from __future__ import annotations

from pathlib import Path


class ProjectRootError(RuntimeError):
    """Raised when the SKU-mapping repository root cannot be identified."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root without depending on the process CWD.

    A valid root contains the packaging manifest and the source package. The
    check intentionally does not require ``.git`` so exported source archives
    remain runnable.
    """
    origin = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    current = origin if origin.is_dir() else origin.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "sku_mapping").is_dir()
        ):
            return candidate
    raise ProjectRootError(f"Could not locate project root from {origin}")


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def project_path(*parts: str) -> Path:
    """Return an absolute path beneath the detected repository root."""
    return PROJECT_ROOT.joinpath(*parts).resolve()


def portable_repository_path(path: str | Path) -> str:
    """Store repository-owned paths relative to the root when possible."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_portable_path(path: str | Path) -> Path:
    """Resolve a stored repository-relative path against the active root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()
