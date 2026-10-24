"""Every reference solution builds a valid part matching its benchmark case."""

import pytest

from eval.bench import load_cases
from eval.references import REFS, check

CASES = {c["id"]: c for c in load_cases()}


def test_references_match_existing_cases():
    assert set(REFS) <= set(CASES)
    assert all(CASES[cid]["expected_bbox_mm"] for cid in REFS)


@pytest.mark.parametrize("cid", sorted(REFS))
def test_reference_is_valid(cid):
    ok, msg = check(cid, CASES)
    assert ok, msg
