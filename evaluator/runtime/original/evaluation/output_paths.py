"""Opt-in output redirection for legacy runners; never redirects source inputs."""
import os
from pathlib import Path

MAIN_ROOT = Path('/__so_arena_runtime__')
WORKTREE_ROOT = Path(__file__).absolute().parents[1]


def output_path(value):
    """Preserve type/defaults; redirect project-relative outputs under JEV_OUTPUT_ROOT.

    Explicit paths outside either project tree remain unchanged. Do not call this
    for input files, model weights, caches, shared locks, or simulator resources.
    """
    configured = os.environ.get('JEV_OUTPUT_ROOT')
    if not configured or value is None:
        return value
    if not isinstance(value, (str, os.PathLike)):
        return value
    path = Path(value).expanduser()
    target = Path(configured).expanduser()
    if not target.is_absolute():
        raise ValueError('JEV_OUTPUT_ROOT must be an absolute path')
    # Do not resolve symlinks: a worktree resource may point to the main tree.
    absolute = Path(os.path.abspath(path))
    try:
        absolute.relative_to(target)
        return value  # Already explicitly isolated; avoid repeated prefixes.
    except ValueError:
        pass
    relative = None
    for root in (WORKTREE_ROOT, MAIN_ROOT):
        try:
            relative = absolute.relative_to(root)
            break
        except ValueError:
            pass
    if relative is None:
        return value
    result = target / relative
    return str(result) if isinstance(value, str) else result


def output_file_path(value):
    """Redirect a direct file-write destination and create its isolated parent."""
    result = output_path(value)
    if os.environ.get('JEV_OUTPUT_ROOT') and result != value:
        Path(result).parent.mkdir(parents=True, exist_ok=True)
    return result
