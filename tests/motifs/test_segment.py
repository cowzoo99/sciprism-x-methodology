def test_short_fragment_merge_respects_protected_unit() -> None:
    from sciprism_x.motifs.segment import segment_trace

    units = segment_trace(
        "Compute acceleration. 9.8 m/s². Therefore it falls.",
        short_fragment_token_threshold=4,
    )

    assert [unit.text for unit in units] == [
        "Compute acceleration.",
        "9.8 m/s².",
        "Therefore it falls.",
    ]


def test_unprotected_short_fragment_is_merged_deterministically() -> None:
    from sciprism_x.motifs.segment import segment_trace

    text = "This holds. The measured pressure remains stable."
    first = segment_trace(text, short_fragment_token_threshold=4)
    second = segment_trace(text, short_fragment_token_threshold=4)

    assert first == second
    assert [unit.text for unit in first] == [
        "This holds. The measured pressure remains stable."
    ]
    assert first[0].start == 0
    assert first[0].end == len(text)


def test_equations_lists_and_discourse_transitions_form_units() -> None:
    from sciprism_x.motifs.segment import segment_trace

    text = "1. Set F = qE. 2. Set F = qvB. Therefore, E = vB."
    units = segment_trace(text, short_fragment_token_threshold=4)

    assert len(units) >= 3
    assert any("qE" in unit.text for unit in units)
    assert any("qvB" in unit.text for unit in units)
    assert any(unit.text.startswith("Therefore") for unit in units)
    assert all(text[unit.start : unit.end].strip() == unit.text for unit in units)
    first_equation = next(unit for unit in units if "qE" in unit.text)
    assert set(first_equation.variables) == {"F", "q", "E"}


def test_articles_are_not_misclassified_as_unit_symbols() -> None:
    from sciprism_x.motifs.segment import segment_trace

    units = segment_trace(
        "A sample remains stable.",
        short_fragment_token_threshold=4,
    )

    assert units[0].units == ()
