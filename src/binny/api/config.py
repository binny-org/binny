"""Survey configuration loader.

This module provides a small helper for reading a survey configuration from a
YAML file and returning it as a :class:`SurveyConfig` container.

The loader enforces a single rule: the YAML document root must be a mapping
(i.e., a dictionary). The returned configuration is stored verbatim in
:attr:`SurveyConfig.raw` so downstream code can interpret keys and structure
as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SurveyConfig:
    """Container for a loaded survey configuration.

    Attributes:
        raw (:no-index:): The parsed YAML content. This is expected to be a
            mapping at the top level (a ``dict[str, Any]``). The schema is
            intentionally not enforced here; validation should be performed by
            callers that require specific keys or value types.
    """

    raw: dict[str, Any]


def load_config(path: str | Path) -> SurveyConfig:
    """Loads a survey configuration from a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        SurveyConfig: An immutable container holding the parsed YAML mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        PermissionError: If the file cannot be opened due to permissions.
        ValueError: If the YAML document root is not a mapping/dictionary.
        yaml.YAMLError: If the YAML file cannot be parsed.

    Examples:
    >>> from pathlib import Path
    >>> import tempfile
    >>> from binny.api.config import load_config
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = Path(d) / "survey.yaml"
    ...     _ = p.write_text("a: 1\\n", encoding="utf-8")
    ...     cfg = load_config(p)
    ...     isinstance(cfg.raw, dict)
    True
    """
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return SurveyConfig(raw=data)
