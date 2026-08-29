import pytest

from dashboard_agent import agent


@pytest.mark.parametrize(
    ("question", "expected_count"),
    [
        (
            "Build exactly three charts: top departments by users; status distribution; monthly enrollments.",
            3,
        ),
        (
            "Create 2 views: compare provinces; then show courses by completion status.",
            2,
        ),
        (
            "Provide multiple visualizations: enrollment trend; certificate breakdown.",
            2,
        ),
    ],
)
def test_multi_view_contract_detects_explicit_dashboard_shape(question, expected_count):
    contract = agent._multi_view_request_contract(question)

    assert contract is not None
    assert contract["requestedCount"] == expected_count
    assert len(contract["clauses"]) >= 2


def test_mixed_language_multi_view_bypasses_lossy_model_canonicalization(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: pytest.fail("multi-view requests must preserve their sibling clauses"),
    )
    question = (
        "\u0e2a\u0e23\u0e49\u0e32\u0e07 3 charts: top departments by distinct users; "
        "status distribution; monthly enrollment split by province."
    )

    canonical = agent._canonicalize_question_for_processing(question, semantic_context={})

    assert "3 charts" in canonical.lower()
    assert "top departments by distinct users" in canonical.lower()
    assert "status distribution" in canonical.lower()
    assert "monthly enrollment split by province" in canonical.lower()


def test_single_chart_request_does_not_create_multi_view_contract():
    assert agent._multi_view_request_contract(
        "Show a chart of enrollments by department split by learning status."
    ) is None


def test_thai_chart_count_preserves_all_analytical_view_terms(monkeypatch):
    monkeypatch.setattr(
        agent,
        "_invoke_language_json",
        lambda **_kwargs: pytest.fail("Thai multi-view structure must not use lossy model rewriting"),
    )
    question = (
        "ช่วยใส่กราฟแยกกัน 3 กราฟในแดชบอร์ดเดียว ได้แก่ 8 หน่วยงานนำแยกตามสถานะผู้เรียน "
        "สัดส่วนสถานะรวม และแนวโน้มลงทะเบียนรายเดือนถึงมิถุนายน 2026 นับคนไม่ซ้ำและใช้ข้อมูลสรุป"
    )

    contract = agent._multi_view_request_contract(question)
    canonical = agent._canonicalize_question_for_processing(question, semantic_context={})

    assert contract is not None
    assert contract["requestedCount"] == 3
    assert "3 chart" in canonical.lower()
    assert "department" in canonical.lower()
    assert "split by learning status" in canonical.lower()
    assert "trend" in canonical.lower()
    assert "monthly" in canonical.lower()
