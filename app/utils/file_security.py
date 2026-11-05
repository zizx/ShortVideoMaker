import os


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # A user-supplied path may be a file name, a relative path, an absolute
    # path, or contain `../`. Resolve it to a real path and use commonpath to
    # check whether it is still inside the allowed directory. This is more
    # reliable than a plain string-prefix check and covers symlinks, repeated
    # separators and relative paths, for allow-listed directories such as the
    # upload, material and task-artifact directories.
    if not unsafe_path:
        raise ValueError("empty path is not allowed")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # On Windows, different drive letters raise ValueError; such a path
        # can never be inside the allowed directory.
        raise ValueError("path is outside the allowed directory") from exc

    if common_path != base_dir_real:
        raise ValueError("path is outside the allowed directory")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("file does not exist")

    return resolved_path
