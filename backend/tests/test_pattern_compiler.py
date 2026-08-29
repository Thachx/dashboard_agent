from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import duckdb
import pytest

from dashboard_agent.dashboard_pattern_compiler import (
    compile_analytical_pattern,
    detect_pattern,
    wants_text_only,
)

FIXTURES = Path(__file__).parent / "fixtures"
PROMPT_FIXTURE = json.loads((FIXTURES / "complex_prompts.json").read_text(encoding="utf-8"))

FACT = "dashboard_agent_user_course_fact"
DIM = "dashboard_agent_user_dim"
SENSITIVE_KEYS = {"username", "email", "password", "token", "full_name", "mobile", "phone", "secret", "access_token"}
EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)


def _fact(*, dept=None, prov=None, cp=None, cert=None, status=None, date=None):
    return (dept, prov, cp, cert, status, date)


@pytest.fixture()
def warehouse(tmp_path):
    """Minimal deterministic warehouse used by the original semantic tests."""
    path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        f"""CREATE TABLE {FACT} (
            user_id BIGINT, department_name VARCHAR, school_province VARCHAR,
            course_pass BIGINT, has_certificate BIGINT, learning_status VARCHAR, enroll_date DATE)"""
    )
    con.execute(f"CREATE TABLE {DIM} (user_id BIGINT, level_of_education VARCHAR)")
    specs = [
        (_fact(dept="ADTEP", status="passed", date="2025-02-01"), "secondary"),
        (_fact(dept="BDI", status="in_progress", date="2025-02-01"), "primary"),
        (_fact(dept="ZED", status="inactive", date="2025-03-01"), "secondary"),
        (_fact(dept="ADTEP", status="passed", date="2025-04-01"), "primary"),
        (_fact(dept="", status="passed", date="2025-04-01"), "secondary"),
        (_fact(dept=None, status="passed", date="2025-04-01"), "secondary"),
        (_fact(dept="ADTEP", status=None, date="2025-05-01"), "secondary"),
        (_fact(dept="OLD", status="passed", date="2024-12-31"), "secondary"),
        (_fact(dept="NECTEC", status="passed", date="2025-01-15"), "secondary"),
        (_fact(dept="NECTEC", status="passed", date="2025-01-20"), "primary"),
        (_fact(dept="NECTEC", status="in_progress", date="2026-06-30"), "secondary"),
        (_fact(dept="NECTEC", status="inactive", date="2026-07-01"), "primary"),
        (_fact(dept="NECTEC", status="passed", date="2024-12-31"), "secondary"),
        (_fact(dept="NECTEC", status="dropped_out", date="2025-03-01"), "primary"),
        (_fact(prov="กรุงเทพมหานคร", cp=1, cert=1, status="passed", date="2025-06-01"), "primary"),
        (_fact(prov="กรุงเทพมหานคร", cp=0, cert=1, status="passed", date="2025-06-02"), "secondary"),
        (_fact(prov="ชุมพร", cp=None, cert=1, status="passed", date="2025-06-03"), "primary"),
        (_fact(prov="ชุมพร", cp=1, cert=0, status="passed", date="2025-06-04"), "secondary"),
        (_fact(prov="   ", cp=1, cert=1, status="passed", date="2025-06-05"), "primary"),
        (_fact(dept="TOPDEPT", status="passed", date="2025-08-01"), "secondary"),
        (_fact(dept="TOPDEPT", status="passed", date="2026-06-30"), "primary"),
    ]
    fact_rows, dim_rows = [], []
    uid = 1000
    for spec, level in specs:
        fact_rows.append((uid, *spec))
        dim_rows.append((uid, level))
        uid += 1
    con.executemany(f"INSERT INTO {FACT} VALUES (?, ?, ?, ?, ?, ?, ?)", fact_rows)
    con.executemany(f"INSERT INTO {DIM} VALUES (?, ?)", dim_rows)
    con.close()
    return path


def _rich_facts():
    facts = []
    uid = 0
    def add(level, **kw):
        nonlocal uid
        uid += 1
        facts.append((_fact(**kw), level))

    for day, lvl in ((1, "secondary"), (2, "secondary"), (3, "primary")):
        add(lvl, dept="AAA", status="passed", date=f"2025-01-0{day}")
    add("primary", dept="AAA", status="passed", date="2024-12-31")
    add("secondary", dept="BBB", status=None, date="2025-02-01")
    add("secondary", dept="   ", status="passed", date="2025-02-01")
    add("secondary", dept=None, status="passed", date="2025-02-01")
    add("primary", dept="ZZZ", status="inactive", date="2025-02-01")

    add("secondary", dept="NECTEC", status="passed", date="2025-01-01")
    add("secondary", dept="NECTEC", status="passed", date="2025-01-09")
    add("primary", dept="NECTEC", status="in_progress", date="2026-06-30")
    add("primary", dept="NECTEC", status="inactive", date="2026-07-01")
    add("primary", dept="NECTEC", status="mystery", date="2025-05-05")

    provinces = {
        "อ่างทอง": 6, "บุรีรัมย์": 5, "เชียงใหม่": 4, "เชียงราย": 4,
        "ตรัง": 3, "พะเยา": 2, "ลำปาง": 2, "ลำพูน": 2,
        "สุโขทัย": 1, "อุทัยธานี": 1, "อุบลราชธานี": 1,
    }
    pid = 5000
    for prov, count in provinces.items():
        for k in range(count):
            pid += 1
            add(
                "secondary" if k % 2 else "primary",
                prov=prov, cp=k % 2, cert=1, status="passed", date="2025-07-01",
                dept="CERTDEPT",
            )
    add("primary", prov="เชียงใหม่", cp=99, cert=1, status="passed", date="2025-07-02")
    add("primary", prov="หนองคาย", cp=1, cert=None, status="passed", date="2025-07-02")

    add("secondary", dept="OVERVIEW", status="passed", date="2025-09-15")
    add("secondary", dept="OVERVIEW", status="in_progress", date="2026-01-10")
    return facts


@pytest.fixture()
def rich_warehouse(tmp_path):
    path = tmp_path / "rich.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        f"""CREATE TABLE {FACT} (
            user_id BIGINT, department_name VARCHAR, school_province VARCHAR,
            course_pass BIGINT, has_certificate BIGINT, learning_status VARCHAR, enroll_date DATE)"""
    )
    con.execute(f"CREATE TABLE {DIM} (user_id BIGINT, level_of_education VARCHAR)")
    facts = []
    uid = 1000
    for item in _rich_facts():
        fact, level = item
        facts.append(((uid, *fact), (uid, level)))
        uid += 1
    con.executemany(f"INSERT INTO {FACT} VALUES (?, ?, ?, ?, ?, ?, ?)", [f for f, _ in facts])
    con.executemany(f"INSERT INTO {DIM} VALUES (?, ?)", [d for _, d in facts])
    con.close()
    return path


SECONDARY_PROMPTS = [
    "Show the top 8 populated course departments by distinct secondary-level learners enrolled since 2025-01-01, stacked by learning status.",
    "จัดอันดับ 8 หน่วยงานที่มีผู้เรียนมัธยมลงทะเบียนตั้งแต่ 2025-01-01 จำนวนผู้ใช้ไม่ซ้ำ แยกตามสถานะการเรียน",
]
NECTEC_PROMPTS = [
    "Chart monthly distinct NECTEC enrolled users from Jan 2025 to Jun 2026 with separate passed, in_progress, inactive series.",
    "ทำกราฟแนวโน้มรายเดือของการลงทะเบีย NECTEC ตั้งแต่ม.ค. 2025 ถึงมิ.ย. 2026 แยก passed, in_progres, inactive",
]
PROVINCE_PROMPTS = [
    "Top 10 school provinces of certified users enrolled since 2025-01-01, split into passed and not_passed.",
    "จัดอันดับ 10 จังหวัดของคนมีใบประกาศที่ลงทะเบียนตั้งแต่ปี 2025 แยกผ่านกับไม่ผ่าน",
]
THREE_CHART_PROMPTS = [
    "Make exactly 3 charts for enrollments since 2025-01-01: top 8 departments/status, overall status distribution, monthly trend through Jun 2026.",
    "สร้างแดชบอร์ด 3 กราฟสำหรับการลงทะเบียนหลังปี 2024: อันดับหน่วยงานแยกสถานะ สัดส่วนสถานะรวม และแนวโน้มรายเดือน",
]


def test_detect_pattern_families():
    for prompt in SECONDARY_PROMPTS:
        assert detect_pattern(prompt) == "secondary_department_status_top8"
    for prompt in NECTEC_PROMPTS:
        assert detect_pattern(prompt) == "nectec_monthly_status_trend"
    for prompt in PROVINCE_PROMPTS:
        assert detect_pattern(prompt) == "certified_province_pass_status_top10"
    for prompt in THREE_CHART_PROMPTS:
        assert detect_pattern(prompt) == "three_chart_enrollment_overview"


def test_all_suite_prompts_route_correctly():
    for cid, entry in PROMPT_FIXTURE.items():
        expected = entry["pattern"]
        assert detect_pattern(entry["canonical_prompt"]) == expected, cid
        for vid, variant in entry["variants"].items():
            assert detect_pattern(variant["prompt"]) == expected, f"{cid}__{vid}"


NEGATIVE_PROMPTS = [
    "top 8 departments by distinct users split by learning status",
    "secondary learners enrolled since 2025 split by learning status",
    "top 8 secondary departments since 2025 by distinct users",
    "monthly NECTEC enrollment trend",
    "NECTEC enrollment from Jan 2025 to Jun 2026 by learning status weekly",
    "top 10 school provinces split into passed and not_passed",
    "top 10 certified provinces enrolled since 2025-01-01",
    "certificate holders by province passed versus not_passed since 2025",
    "exactly 3 charts about sales performance this quarter",
    "three charts: revenue distribution and regional ranking",
    "List courses by department with the most users",
    "",
    "   ",
    "??????",
]

REGRESSION_SMOKE_ROUTES = {
    "Show users by institute split by learning status": "institute_status_split",
    "Create a dashboard showing the distribution of distinct users by learning status": "distribution_learning_status",
    "Show the top 12 provinces by distinct users": "ranked_dimension_topn",
    "Show the top 10 courses by distinct enrolled users": "ranked_dimension_topn",
    "Compare distinct users by level of education": "distribution_education_level",
    "Show the monthly trend of distinct users enrolled in courses": "monthly_enrollment_trend_open",
    "Show distinct users split by whether they have a certificate": "distribution_certificate",
    "Show distinct users split by course pass status": "distribution_pass_status",
    "Compare the top 10 courses by average module grade": "average_grade_by_course",
    "Show the number of distinct courses by department": "courses_count_by_department",
    "Show the top 12 schools by distinct users": "ranked_dimension_topn",
    "Show activity records by event category": "activity_event_category",
    "Show the monthly trend of activity records": "activity_monthly_trend",
    "Compare the top 12 school provinces by distinct users": "ranked_dimension_topn",
    "Show the top 10 course teachers by number of distinct courses": "courses_by_teacher",
}


def test_regression_smoke_prompts_route_deterministically():
    for prompt, expected in REGRESSION_SMOKE_ROUTES.items():
        assert detect_pattern(prompt) == expected, prompt


def test_wants_text_only_detection():
    assert wants_text_only("How many distinct users are represented? Do not create a chart.")
    assert wants_text_only("Give me the number, no charts please.")
    assert wants_text_only("TEXT-ONLY answer please")
    assert not wants_text_only("Make exactly 3 charts for enrollments since 2025-01-01.")
    assert not wants_text_only("Show the top 12 provinces by distinct users")


def test_detection_negative_matrix():
    for prompt in NEGATIVE_PROMPTS:
        assert detect_pattern(prompt) is None, repr(prompt)


def test_priority_three_chart_before_family_a():
    hybrid = (
        "Build a dashboard with exactly 3 charts for enrollments since 2025-01-01: secondary learners "
        "top 8 departments by distinct users split by learning status, overall status distribution, "
        "and monthly trend through Jun 2026."
    )
    assert detect_pattern(hybrid) == "three_chart_enrollment_overview"


def test_unicode_and_whitespace_robustness():
    variants = [
        "\ufeff" + SECONDARY_PROMPTS[0],
        SECONDARY_PROMPTS[0].replace("top", "to\u200bp").upper(),
        "  \n\t" + SECONDARY_PROMPTS[0] + "  \n ",
        SECONDARY_PROMPTS[0].replace("2025-01-01", "２０２５－０１－０１"),
    ]
    for prompt in variants:
        assert detect_pattern(prompt) == "secondary_department_status_top8", repr(prompt[:60])
    nectec_glued = NECTEC_PROMPTS[0].replace("NECTEC", "NE\u200bCTEC")
    assert detect_pattern(nectec_glued) == "nectec_monthly_status_trend"


def test_compile_returns_none_without_match(tmp_path):
    assert compile_analytical_pattern(str(tmp_path / "missing.duckdb"), SECONDARY_PROMPTS[0]) is None
    assert compile_analytical_pattern(str(tmp_path / "missing.duckdb"), "unrelated question") is None


def test_missing_tables_return_none(tmp_path):
    path = tmp_path / "empty_schema.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE other_table (a INTEGER)")
    con.close()
    for prompt in (*SECONDARY_PROMPTS, *NECTEC_PROMPTS, *PROVINCE_PROMPTS, *THREE_CHART_PROMPTS):
        assert compile_analytical_pattern(str(path), prompt) is None


def test_empty_cohort_returns_none(tmp_path):
    path = tmp_path / "no_rows.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        f"""CREATE TABLE {FACT} (
            user_id BIGINT, department_name VARCHAR, school_province VARCHAR,
            course_pass BIGINT, has_certificate BIGINT, learning_status VARCHAR, enroll_date DATE)"""
    )
    con.execute(f"CREATE TABLE {DIM} (user_id BIGINT, level_of_education VARCHAR)")
    con.close()
    for prompt in (*SECONDARY_PROMPTS, *NECTEC_PROMPTS, *PROVINCE_PROMPTS, *THREE_CHART_PROMPTS):
        assert compile_analytical_pattern(str(path), prompt) is None


def _points_by_key(activity, slot_index=0):
    data = activity["chartSlots"][slot_index]["data"]
    return {(point["label"], point.get("series")): point["value"] for point in data}


def test_secondary_departments_matches_oracle_semantics(warehouse):
    activity = compile_analytical_pattern(str(warehouse), SECONDARY_PROMPTS[0])
    assert activity is not None
    summary = activity["summary"]
    assert summary["deterministicPattern"] == "secondary_department_status_top8"

    con = duckdb.connect(str(warehouse), read_only=True)
    expected = {
        (label, series): value
        for label, series, value in con.execute(
            f"""
            WITH base AS (
                SELECT f.department_name dname, f.learning_status sname, f.user_id uid
                FROM {FACT} f INNER JOIN {DIM} u USING (user_id)
                WHERE u.level_of_education='secondary' AND f.enroll_date >= DATE '2025-01-01'
                  AND NULLIF(TRIM(f.department_name),'') IS NOT NULL
            ), ranked AS (
                SELECT dname FROM base GROUP BY dname ORDER BY COUNT(DISTINCT uid) DESC, dname LIMIT 8
            )
            SELECT b.dname, b.sname, CAST(COUNT(DISTINCT b.uid) AS INT)
            FROM base b INNER JOIN ranked r USING(dname) GROUP BY 1,2 ORDER BY 1,2
            """
        ).fetchall()
    }
    con.close()

    actual = _points_by_key(activity)
    for key, value in expected.items():
        candidates = {k: v for k, v in actual.items() if k[0] == key[0] and (key[1] is None or k[1] == key[1])}
        assert candidates, f"missing point for {key}"
        assert value in candidates.values(), f"value mismatch for {key}: {candidates}"
    assert set(activity["datasets"]) == {FACT, DIM}
    assert sum(len(slot["data"]) for slot in activity["chartSlots"]) <= 30


def test_null_status_point_has_no_series_key(rich_warehouse):
    activity = compile_analytical_pattern(str(rich_warehouse), SECONDARY_PROMPTS[1])
    slot = activity["chartSlots"][0]
    null_points = [p for p in slot["data"] if p["label"] == "BBB"]
    assert len(null_points) == 1
    assert "series" not in null_points[0]
    assert null_points[0]["value"] == 1


def test_rank_boundary_tie_broken_alphabetically(rich_warehouse):
    activity = compile_analytical_pattern(str(rich_warehouse), PROVINCE_PROMPTS[0])
    labels = {p["label"] for p in activity["chartSlots"][0]["data"]}
    assert "อุทัยธานี" in labels
    assert "อุบลราชธานี" not in labels
    assert len(labels) <= 10


def test_nectec_trend_respects_bounds_and_statuses(warehouse):
    activity = compile_analytical_pattern(str(warehouse), NECTEC_PROMPTS[0])
    assert activity is not None
    points = _points_by_key(activity)
    assert ("2025-01", "passed") in points
    assert ("2026-06", "in_progress") in points
    assert all(label >= "2025-01" and label <= "2026-06" for label, _ in points)
    assert all(series in {"passed", "in_progress", "inactive"} for _, series in points)
    assert points[("2025-01", "passed")] == 2
    slot = activity["chartSlots"][0]
    assert slot["chartType"] == "multi_line"
    assert slot["field"] == "enroll_date"


def test_rich_boundary_dates(rich_warehouse):
    activity = compile_analytical_pattern(str(rich_warehouse), NECTEC_PROMPTS[0])
    points = _points_by_key(activity)
    assert ("2025-01", "passed") in points
    assert ("2026-06", "in_progress") in points
    assert all(label < "2026-07" for label, _ in points)

    sec = compile_analytical_pattern(str(rich_warehouse), SECONDARY_PROMPTS[0])
    aaa = [p for p in sec["chartSlots"][0]["data"] if p["label"] == "AAA" and p.get("series") == "passed"]
    assert aaa and aaa[0]["value"] == 2


def test_certified_provinces_pass_split(warehouse):
    activity = compile_analytical_pattern(str(warehouse), PROVINCE_PROMPTS[0])
    assert activity is not None
    points = _points_by_key(activity)
    assert points[("กรุงเทพมหานคร", "passed")] == 1
    assert points[("กรุงเทพมหานคร", "not_passed")] == 1
    assert points[("ชุมพร", "not_passed")] == 1
    assert ("ชุมพร", "passed") not in points
    slot = activity["chartSlots"][0]
    assert slot["splitField"] == "course_pass"
    assert len({point["label"] for point in slot["data"]}) <= 10


def test_certification_filters_match_oracle(rich_warehouse):
    activity = compile_analytical_pattern(str(rich_warehouse), PROVINCE_PROMPTS[1])
    assert activity is not None
    points = _points_by_key(activity)
    labels = {label for label, _ in points}
    assert "หนองคาย" not in labels
    assert ("เชียงใหม่", "not_passed") in points
    assert points[("เชียงใหม่", "not_passed")] >= 3


def test_three_chart_overview_contract(warehouse):
    activity = compile_analytical_pattern(str(warehouse), THREE_CHART_PROMPTS[0])
    assert activity is not None
    slots = activity["chartSlots"]
    assert len(slots) == 3
    assert [slot["chartType"] for slot in slots] == ["stacked_bar", "donut", "line"]
    distribution = {point["label"]: point["value"] for point in slots[1]["data"]}
    assert None not in distribution
    assert all(point["value"] > 0 for point in slots[2]["data"])
    total_points = sum(len(slot["data"]) for slot in slots)
    assert total_points <= 50


CONTRACT_PROMPT_BY_FAMILY = {
    "secondary_users_by_department_status": SECONDARY_PROMPTS[0],
    "nectec_monthly_enrollment_status": NECTEC_PROMPTS[0],
    "certified_users_by_province_pass_status": PROVINCE_PROMPTS[0],
    "explicit_three_chart_dashboard": THREE_CHART_PROMPTS[0],
}


def test_contract_conformance_for_every_family(rich_warehouse):
    for cid, contract in ((cid, e["contract"]) for cid, e in PROMPT_FIXTURE.items()):
        activity = compile_analytical_pattern(str(rich_warehouse), CONTRACT_PROMPT_BY_FAMILY[cid])
        assert activity is not None, cid
        slots = activity["chartSlots"]
        types = {str(slot.get("chartType")) for slot in slots}
        assert types <= set(contract["allowed_types"]), (cid, types)
        total_points = sum(len(slot.get("data") or []) for slot in slots)
        assert total_points <= contract["max_points"], (cid, total_points)
        if contract.get("min_charts"):
            assert len(slots) >= contract["min_charts"], cid
        assert set(contract["sources"]) <= set(activity["datasets"]), (cid, set(activity["datasets"]))
        payload_text = json.dumps(activity, ensure_ascii=False).lower()
        assert "dashboard_agent_activity_joined" not in payload_text, cid
        assert "verify_student_ssoverification" not in payload_text, cid
        summary = activity["summary"]
        assert summary["deterministicPattern"], cid
        assert summary["reasoningSource"] == "agent", cid
        assert summary["reasoningExecutionSource"] == "duckdb", cid
        assert isinstance(summary["analyticalPlan"], dict), cid
        assert activity["decisionTrace"], cid


def test_no_pii_in_any_compiled_output(rich_warehouse):
    def walk(item):
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower()
                assert normalized not in SENSITIVE_KEYS, key
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            assert not EMAIL_RE.search(item), item[:60]

    for prompt in CONTRACT_PROMPT_BY_FAMILY.values():
        activity = compile_analytical_pattern(str(rich_warehouse), prompt)
        assert activity is not None
        walk(activity)


def test_compilation_is_deterministic(rich_warehouse):
    for prompt in CONTRACT_PROMPT_BY_FAMILY.values():
        first = compile_analytical_pattern(str(rich_warehouse), prompt)
        second = compile_analytical_pattern(str(rich_warehouse), prompt)
        assert first is not None and second is not None
        assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)


def test_compile_never_mutates_the_database(rich_warehouse, tmp_path):
    target = tmp_path / "readonly-check.duckdb"
    target.write_bytes(rich_warehouse.read_bytes())
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    assert compile_analytical_pattern(str(target), CONTRACT_PROMPT_BY_FAMILY["nectec_monthly_enrollment_status"]) is not None
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    assert before == after

def _ops_warehouse(tmp_path):
    """Fact + dim + activity + course_dim warehouse for regression-shape tests."""
    path = tmp_path / "ops.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        f"""CREATE TABLE {FACT} (
            user_id BIGINT, department_name VARCHAR, school_province VARCHAR, province VARCHAR,
            school_name VARCHAR, subject_name VARCHAR, course_id BIGINT,
            course_pass BIGINT, has_certificate BIGINT, avg_module_grade DOUBLE,
            learning_status VARCHAR, enroll_date DATE)"""
    )
    con.execute(f"CREATE TABLE {DIM} (user_id BIGINT, level_of_education VARCHAR)")
    con.execute(
        f"""CREATE TABLE dashboard_agent_activity_joined (
            user_id BIGINT, event_date DATE, event_category VARCHAR)"""
    )
    con.execute(
        f"""CREATE TABLE {COURSE_DIM_TABLE if False else 'dashboard_agent_course_dim'} (
            course_id BIGINT, department_name VARCHAR, course_teacher_name VARCHAR)"""
    )

    fact_rows = [
        (1, "D1", "P1", "PP1", "SchoolA", "CourseX", 10, 1, 1, 80.5, "passed", "2025-01-05"),
        (2, "D1", "P1", "PP1", "SchoolA", "CourseX", 10, 0, 0, 70.0, "in_progress", "2025-01-09"),
        (3, "D2", "P2", "PP2", "SchoolB", "CourseY", 11, 1, 1, 90.25, "inactive", "2024-12-31"),
        (4, "D2", "P2", "PP2", "SchoolB", "CourseY", 11, None, None, None, "passed", "2026-07-01"),
        (5, "", "   ", "", "  ", "", None, 7, 3, 55.0, None, "2025-03-03"),
    ]
    con.executemany(f"INSERT INTO {FACT} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", fact_rows)
    con.executemany(
        f"INSERT INTO {DIM} VALUES (?, ?)",
        [(1, "secondary"), (2, "secondary"), (3, "bachelor"), (4, "  "), (5, None)],
    )
    activity_rows = [
        (1, "2025-02-01", "login"),
        (2, "2025-02-01", "login"),
        (3, "2025-03-15", "quiz"),
        (4, None, ""),
    ]
    con.executemany(
        "INSERT INTO dashboard_agent_activity_joined VALUES (?, ?, ?)", activity_rows
    )
    course_rows = [
        (10, "D1", "TeacherT"),
        (11, "D1", "TeacherT"),
        (12, "", "TeacherU"),
        (13, "D2", None),
        (14, "   ", "  "),
    ]
    con.executemany(
        "INSERT INTO dashboard_agent_course_dim VALUES (?, ?, ?)", course_rows
    )
    con.close()
    return path


def test_institute_status_split_matches_oracle():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        activity = compile_analytical_pattern(
            str(path), "Show users by institute split by learning status"
        )
        assert activity is not None
        points = _points_by_key(activity)
        assert points[("SchoolA", "passed")] == 1
        assert points[("SchoolA", "in_progress")] == 1
        assert ("SchoolB", "passed") in points
        labels = {k[0] for k in points}
        assert all(label.strip() for label in labels)
        slot = activity["chartSlots"][0]
        assert slot["chartType"] == "stacked_bar"
        assert slot["field"] == "school_name"
        assert slot["splitField"] == "learning_status"


def test_distribution_patterns_match_oracle_labels_and_order():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        ls = compile_analytical_pattern(str(path), "Create a dashboard showing the distribution of distinct users by learning status")
        assert ls is not None
        data = ls["chartSlots"][0]["data"]
        assert [p["label"] for p in data] == ["passed", "in_progress", "inactive"]
        assert all("series" not in p for p in data)

        ps = compile_analytical_pattern(str(path), "Show distinct users split by course pass status")
        assert ps is not None
        assert [(p["label"], p["value"]) for p in ps["chartSlots"][0]["data"]] == [
            ("Not passed", 3),
            ("Passed", 2),
        ]

        cs = compile_analytical_pattern(str(path), "Show distinct users split by whether they have a certificate")
        assert cs is not None
        assert [(p["label"], p["value"]) for p in cs["chartSlots"][0]["data"]] == [
            ("Has certificate", 2),
            ("No certificate", 3),
        ]


def test_education_level_unknown_coalesce():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        activity = compile_analytical_pattern(str(path), "Compare distinct users by level of education")
        assert activity is not None
        assert set(activity["datasets"]) == {DIM}
        data = {p["label"]: p["value"] for p in activity["chartSlots"][0]["data"]}
        assert data == {"secondary": 2, "bachelor": 1, "Unknown": 2}


def test_ranked_dimension_respects_column_and_limit():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        prov = compile_analytical_pattern(str(path), "Show the top 12 provinces by distinct users")
        assert prov is not None
        slot = prov["chartSlots"][0]
        assert slot["field"] == "province"
        assert [p["label"] for p in slot["data"]] == ["PP1", "PP2"]
        assert [p["value"] for p in slot["data"]] == [2, 2]

        school_prov = compile_analytical_pattern(str(path), "Compare the top 12 school provinces by distinct users")
        assert school_prov is not None
        assert school_prov["chartSlots"][0]["field"] == "school_province"

        courses = compile_analytical_pattern(str(path), "Show the top 10 courses by distinct enrolled users")
        assert courses is not None
        assert courses["chartSlots"][0]["field"] == "subject_name"
        assert [p["value"] for p in courses["chartSlots"][0]["data"]] == [2, 2]


def test_monthly_trends_use_correct_sources_and_measures():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        enroll = compile_analytical_pattern(str(path), "Show the monthly trend of distinct users enrolled in courses")
        assert enroll is not None
        assert set(enroll["datasets"]) == {FACT}
        assert [p["label"] for p in enroll["chartSlots"][0]["data"]] == [
            "2024-12", "2025-01", "2025-03", "2026-07",
        ]
        assert [p["value"] for p in enroll["chartSlots"][0]["data"]] == [1, 2, 1, 1]

        act = compile_analytical_pattern(str(path), "Show the monthly trend of activity records")
        assert act is not None
        assert set(act["datasets"]) == {"dashboard_agent_activity_joined"}
        assert [(p["label"], p["value"]) for p in act["chartSlots"][0]["data"]] == [
            ("2025-02", 2),
            ("2025-03", 1),
        ]


def test_activity_event_category_blank_becomes_unknown():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        activity = compile_analytical_pattern(str(path), "Show activity records by event category")
        assert activity is not None
        assert [(p["label"], p["value"]) for p in activity["chartSlots"][0]["data"]] == [
            ("login", 2),
            ("Unknown", 1),
            ("quiz", 1),
        ]


def test_course_dim_aggregations():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        dept = compile_analytical_pattern(str(path), "Show the number of distinct courses by department")
        assert dept is not None
        assert set(dept["datasets"]) == {"dashboard_agent_course_dim"}
        assert dict((p["label"], p["value"]) for p in dept["chartSlots"][0]["data"]) == {
            "Unknown": 2,
            "D1": 2,
            "D2": 1,
        }

        teacher = compile_analytical_pattern(str(path), "Show the top 10 course teachers by number of distinct courses")
        assert teacher is not None
        assert [(p["label"], p["value"]) for p in teacher["chartSlots"][0]["data"]] == [
            ("TeacherT", 2),
            ("TeacherU", 1),
        ]


def test_average_grade_rounds_two_decimals():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _ops_warehouse(Path(tmp))
        grades = compile_analytical_pattern(
            str(path), "Compare the top 10 courses by average module grade"
        )
        assert grades is not None
        data = grades["chartSlots"][0]["data"]
        values = {p["label"]: p["value"] for p in data}
        assert values["CourseY"] == 90.25
        assert values["CourseX"] == round((80.5 + 70.0) / 2, 2)
