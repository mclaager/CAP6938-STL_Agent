"""Retrieval methods (word and letter-chunk TF-IDF)."""

import pytest

from stl_agent.retrieval import CharTfidfRetriever, make_retriever, strip_numbers


def test_strip_numbers_drops_sizes_only():
    assert strip_numbers("A 60mm x 40mm plate, 4.5 mm holes at 90 degrees").split() == [
        "a", "plate,", "holes", "at"]


def test_char_tfidf_matches_word_variants():
    r = CharTfidfRetriever.from_file()
    assert r.retrieve("A hexagonal prism 30mm across the flats and 8mm tall.", k=1)[0]["id"] == "polygon_001"
    assert r.retrieve("A ruler with my name engraved on it", k=1)[0]["cls"] == "text"


def test_make_retriever_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_retriever("nope")
