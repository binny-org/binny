"""Utilities for validating comparison relations used in filters."""

from typing import Literal

Relation = Literal["lt", "le", "gt", "ge"]

_RELATIONS: dict[str, str] = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


def available_relations() -> list[str]:
    """Return supported relation identifiers."""
    return sorted(_RELATIONS)


def validate_relation(name: str) -> Relation:
    """Validate a relation string.

    Raises:
        ValueError: If the relation is not supported.
    """
    if name not in _RELATIONS:
        allowed = ", ".join(sorted(_RELATIONS))
        raise ValueError(f"Unknown relation {name!r}. Allowed relations are: {allowed}.")
    return name  # type: ignore
