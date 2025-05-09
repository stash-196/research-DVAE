import sys
from pathlib import Path


def find_project_root(current_file) -> Path:
    """
    Traverse up the directory tree from the current file to find the project root
    by detecting the `src/` directory.

    Args:
        current_file (str | Path): The path of the current file (__file__).

    Returns:
        Path: The Path object representing the project root directory.

    Raises:
        FileNotFoundError: If the `src/` directory is not found in the path hierarchy.
    """
    current_path = Path(current_file).resolve()  # Ensure it's a Path object

    for parent in [current_path] + list(current_path.parents):
        if (parent / "src").is_dir():
            return parent  # Project root is the parent of `src/`

    raise FileNotFoundError("Could not find project root containing 'src/' directory.")


def setup_project_root():
    """
    Detect the project root and insert it into sys.path.
    """
    try:
        project_root = find_project_root(__file__)
        project_root_str = str(project_root)

        if project_root_str not in sys.path:
            print(f"[INFO] Project root detected at: {project_root_str}")
            sys.path.insert(0, project_root_str)
            print(f"[INFO] Project root added to sys.path")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
