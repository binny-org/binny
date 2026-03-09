"""Unit tests for binny.utils.relations."""

from __future__ import annotations

import pytest

import binny.utils.relations as rel


def test_available_relations_returns_sorted_list():
    """Tests that available_relations returns supported relations in sorted order."""
    out = rel.available_relations()
    assert out == ["ge", "gt", "le", "lt"]


@pytest.mark.parametrize("name", ["lt", "le", "gt", "ge"])
def test_validate_relation_accepts_valid_relations(name):
    """Tests that validate_relation returns valid relation names unchanged."""
    assert rel.validate_relation(name) == name


def test_validate_relation_rejects_unknown_relation():
    """Tests that validate_relation raises ValueError for unsupported relations."""
    with pytest.raises(ValueError, match=r"Unknown relation"):
        rel.validate_relation("eq")


def test_validate_relation_error_message_lists_allowed_relations():
    """Tests that the error message lists allowed relations."""
    with pytest.raises(ValueError) as exc:
        rel.validate_relation("bad")

    msg = str(exc.value)
    assert "Allowed relations" in msg
    assert "lt" in msg
    assert "le" in msg
    assert "gt" in msg
    assert "ge" in msg
