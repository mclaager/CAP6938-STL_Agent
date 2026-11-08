"""Statistics helpers (eval/stats.py): case-level bootstrap, sign test, kappa, McNemar."""

import pytest

from eval import stats


def test_per_case_rates_average_repeats():
    rows = [dict(case="a", correct=True), dict(case="a", correct=False), dict(case="b", correct=True),
            dict(case="c", correct=None)]
    assert stats.per_case(rows, lambda r: r["correct"]) == {"a": 0.5, "b": 1.0}


def test_bootstrap_resamples_cases_and_is_reproducible():
    mean, lo, hi = stats.bootstrap_mean([1, 1, 1, 0, 0.5] * 4)
    assert mean == pytest.approx(0.7) and lo < mean < hi
    assert stats.bootstrap_mean([1, 1, 1, 0, 0.5] * 4) == (mean, lo, hi)  # fixed seed
    assert stats.bootstrap_mean([1.0] * 10) == (1.0, 1.0, 1.0)
    narrow = stats.bootstrap_mean([1, 0] * 50)
    wide = stats.bootstrap_mean([1, 0] * 5)
    assert narrow[2] - narrow[1] < wide[2] - wide[1]  # more cases, narrower interval


def test_sign_test_matches_the_rag_copying_study():
    assert stats.sign_test(8, 10) == pytest.approx(0.8145, abs=1e-4)  # rag-copying.md: p = 0.82
    assert stats.sign_test(0, 0) == 1.0
    assert stats.sign_test(10, 0) == pytest.approx(2 / 1024)


def test_paired_uses_shared_cases_only():
    a = {"x": 1.0, "y": 1.0, "z": 0.0, "only_a": 1.0}
    b = {"x": 0.0, "y": 1.0, "z": 0.0}
    p = stats.paired(a, b)
    assert (p["n"], p["better"], p["worse"], p["tied"]) == (3, 1, 0, 2)
    assert p["diff"] == pytest.approx(1 / 3) and p["p"] == 1.0
    assert stats.fmt_diff(p).startswith("+33 pts")


def test_kappa_and_mcnemar():
    assert stats.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert stats.cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    bands = ["match", "minor", "feature_wrong", "different"]
    near = stats.cohen_kappa(["match", "minor", "different", "feature_wrong"], ["minor", "match", "feature_wrong",
                                                                                 "different"], bands, "linear")
    far = stats.cohen_kappa(["match", "minor", "different", "feature_wrong"], ["different", "feature_wrong",
                                                                                "match", "minor"], bands, "linear")
    assert near > far  # one band off counts as partly agreeing
    m = stats.mcnemar([True, True, False, True], [False, True, False, False])
    assert (m["only_a"], m["only_b"], m["p"]) == (2, 0, 0.5)


def test_quantiles_and_formatting():
    assert stats.quantiles([1, 2, 3, None]) == (2.0, pytest.approx(2.8))
    assert stats.quantiles([]) == (None, None)
    assert stats.fmt_rate(0.857, 0.8, 0.91) == "86% [80-91]"


def _row(model, config, case, rep, correct, outcome="built", **kw):
    return dict(dict(model=model, config=config, case=case, rep=rep, library="lib1", category="simple",
                     expect="build", correct=correct, built=outcome == "built", outcome=outcome, first_try=correct,
                     requests=1, seconds=10.0, llm_seconds=6.0, wait_seconds=1.0, sandbox_seconds=2.0,
                     output_tokens=600, usd=0.01, study="s1", attempts=1, failures=[]), **kw)


def test_study_report_tables_and_comparisons():
    from eval import study
    from eval.bench import part_id

    rows = [_row("m", "rag-size", c, n, ok) for c, oks in (("a", [1, 1]), ("b", [1, 0]), ("c", [0, 0]))
            for n, ok in enumerate(oks)]
    rows += [_row("m", "norag-size", c, n, ok) for c, oks in (("a", [0, 0]), ("b", [1, 0]), ("c", [0, 0]))
             for n, ok in enumerate(oks)]
    judge_rows = [dict(kind="bench", part=part_id(r), score=0.9 if r["case"] == "a" else 0.1, seconds=4.0)
                  for r in rows if r["correct"]]
    verdicts = study.judge_verdicts(judge_rows, threshold=0.5)
    assert study.shape_ok(rows[0], verdicts) is True and study.shape_ok(rows[2], verdicts) is False
    report = study.study_report(rows, verdicts, [dict(name="rag", a="m:rag-size", b="m:norag-size")], "j")
    assert "| m | main | rag-size | 3 | 6 | 50% [" in report  # (1 + 0.5 + 0) / 3 cases
    assert "| m | main | norag-size | 3 | 6 | 17% [" in report
    assert "| rag | correct | 3 | 50% | 17% | +33 pts" in report and "| 1 / 0 / 2 | 1.000 |" in report
    assert "| m | main | rag-size | 10 (10) | 6.0 (6.0) | 1.0 (1.0) | 2.0 (2.0) | 100 | 4.0 (4.0) | 14 (14) |" in report
    # 6 runs of 600 output tokens (no input tokens in these rows), 3 correct: 1,200 tokens and $0.02 per correct part
    assert "| m | main | rag-size | 0 | 600 | 1,200 | $0.0200 |" in report


def test_organic_shape_counts_any_part_the_judge_passes():
    from eval import study
    from eval.bench import part_id

    leaky = _row("thingiverse", "thingiverse-check", "o13", 0, False, outcome="best_effort", category="organic")
    nothing = _row("thingiverse", "thingiverse-check", "o14", 0, False, outcome="failed", category="organic")
    assert study.shape_ok(leaky, {part_id(leaky): {"pass": True, "seconds": 1}}) is True
    assert study.shape_ok(leaky, {}) is None and study.shape_ok(nothing, {}) is False
    main = _row("m", "rag-size", "m01", 0, False, outcome="best_effort")
    assert study.shape_ok(main, {part_id(main): {"pass": True, "seconds": 1}}) is False  # main suite: size must pass


def test_organic_suite_can_use_another_judge():
    from eval import study
    from eval.bench import part_id

    rows = [_row("m", "sdf", f"o{i}", 0, True, category="organic") for i in range(4)]
    main = {part_id(r): {"pass": False, "seconds": 1.0} for r in rows}
    organic = {part_id(r): {"pass": True, "seconds": 2.0} for r in rows}
    report = study.study_report(rows, main, judge_name="A", organic_verdicts=organic, organic_judge_name="B")
    assert "| m | organic | sdf | 4 | 4 | 100% [100-100] | 100% [100-100] | 4 |" in report  # B's verdicts count
    assert "| m | sdf | 100% [100-100] | 0% [0-0] |" in report  # both judges side by side
    assert "(A; organic suite: B)" in report


def test_repair_and_retry_tables():
    from eval import study

    assert study.pass_at_k(3, 1, 1) == pytest.approx(1 / 3) and study.pass_at_k(3, 1, 3) == 1.0
    assert study.pass_at_k(3, 0, 2) == 0.0 and study.pass_at_k(3, 2, 2) == 1.0
    rows = [_row("m", "rag-size", "a", 0, True, attempts=1, failures=[]),
            _row("m", "rag-size", "a", 1, True, attempts=3, failures=["exec_error", "invalid_mesh"]),
            _row("m", "rag-size", "a", 2, False, outcome="best_effort", attempts=4,
                 failures=["exec_error", "exec_error", "invalid_mesh", "invalid_mesh"])]
    f = study.fixed_next(rows)
    assert f["exec_error"] == [3, 0] and f["invalid_mesh"] == [3, 1]  # only the mesh failure before an ok was fixed
    out = "\n".join(study.repair_tables(rows))
    assert "| m | main | rag-size | 33% [33-33] | 33% | 67% | 67% [67-67] | 50% of 2 |" in out
    assert "| m | main | rag-size | 0% of 3 | 33% of 3 | - | - |" in out
    assert "| m | main | rag-size | 67% [67-67] | 100% [100-100] | 100% [100-100] |" in out
