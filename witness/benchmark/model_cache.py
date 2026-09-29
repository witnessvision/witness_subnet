"""Delete explicitly retired model directories, never evaluation evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys


def remove_models(root: Path, models: list[str]) -> list[str]:
    """Only named hash directories below a real cache root may be removed."""
    if any(not isinstance(model, str) or not re.fullmatch('[0-9a-f]{64}', model) for model in models):
        raise ValueError('invalid_model_cache_id')
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError('model_cache_symlink')
    if not root.exists():
        return []
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError('safe_model_cache_removal_unavailable')
    removed = []
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for model in sorted(set(models)):
            try:
                mode = os.stat(model, dir_fd=fd, follow_symlinks=False).st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(mode):
                raise ValueError('unsafe_model_cache_directory')
            # fd-based rmtree does not follow symlinks inside the tree either.
            shutil.rmtree(model, dir_fd=fd)
            removed.append(model)
    finally:
        os.close(fd)
    return removed


if __name__ == '__main__':
    print(json.dumps(remove_models(Path(sys.argv[1]), sys.argv[2:])))
