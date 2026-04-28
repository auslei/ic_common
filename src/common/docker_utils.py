"""
Utility for resolving project paths across host and Docker environments.

Docker runtime is detected by the presence of /.dockerenv or /app/projects.
Projects live at:
  - Host:   projects/{project_name}/
  - Docker: /app/projects/{project_name}/
"""

from pathlib import Path


def is_docker_runtime() -> bool:
    """Return True when running inside a Docker container."""
    return Path("/.dockerenv").exists() or Path("/app/projects").exists()


def resolve_project_base() -> Path:
    """Return the root projects directory for the current runtime environment."""
    return Path("/app/projects") if is_docker_runtime() else Path("projects")


def resolve_project_dir(project_name: str) -> Path:
    """Return the top-level directory for a named project."""
    return resolve_project_base() / project_name
