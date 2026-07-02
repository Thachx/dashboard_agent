#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


DEFAULT_DATABASE = Path("/home/thacha/dashboard_agent/data/_warehouse/dashboard_agent.duckdb")


def log(message: str) -> None:
    print(message, flush=True)


def build_joined_mart(database: Path) -> None:
    started = time.time()
    con = duckdb.connect(str(database))
    try:
        log("creating dashboard_agent_user_dim")
        con.execute(
            """
            create or replace table dashboard_agent_user_dim as
            select
                try_cast(json_extract_string(payload_json, '$.user_id') as bigint) as user_id,
                json_extract_string(payload_json, '$.username') as username,
                json_extract_string(payload_json, '$.email') as email,
                json_extract_string(payload_json, '$.full_name') as full_name,
                json_extract_string(payload_json, '$.institute_id') as institute_id,
                json_extract_string(payload_json, '$.school_name') as school_name,
                json_extract_string(payload_json, '$.school_province') as school_province,
                json_extract_string(payload_json, '$.level_of_education') as level_of_education,
                try_cast(json_extract_string(payload_json, '$.date_joined') as timestamp) as date_joined,
                try_cast(json_extract_string(payload_json, '$.last_login') as timestamp) as last_login,
                payload_json as user_payload_json
            from unified_records
            where source_path = 'raw-parquet/dim_user.parquet'
              and payload_json is not null
            qualify row_number() over (
                partition by try_cast(json_extract_string(payload_json, '$.user_id') as bigint)
                order by record_index
            ) = 1
            """
        )

        log("creating dashboard_agent_user_course_fact")
        con.execute(
            """
            create or replace table dashboard_agent_user_course_fact as
            select
                try_cast(json_extract_string(payload_json, '$.user_id') as bigint) as user_id,
                json_extract_string(payload_json, '$.course_id') as course_id,
                json_extract_string(payload_json, '$.username') as username,
                json_extract_string(payload_json, '$.email') as email,
                json_extract_string(payload_json, '$.full_name') as full_name,
                json_extract_string(payload_json, '$.province') as province,
                json_extract_string(payload_json, '$.school_name') as school_name,
                json_extract_string(payload_json, '$.school_province') as school_province,
                json_extract_string(payload_json, '$.subject_name') as subject_name,
                json_extract_string(payload_json, '$.department_name') as department_name,
                json_extract_string(payload_json, '$.course_type') as course_type,
                json_extract_string(payload_json, '$.course_org_name') as course_org_name,
                json_extract_string(payload_json, '$.course_faculty_name') as course_faculty_name,
                json_extract_string(payload_json, '$.course_teacher_name') as course_teacher_name,
                try_cast(json_extract_string(payload_json, '$.enroll_date') as timestamp) as enroll_date,
                try_cast(json_extract_string(payload_json, '$.activity_count') as double) as course_activity_count,
                try_cast(json_extract_string(payload_json, '$.avg_module_grade') as double) as avg_module_grade,
                try_cast(json_extract_string(payload_json, '$.max_module_grade') as double) as max_module_grade,
                try_cast(json_extract_string(payload_json, '$.last_activity_date') as timestamp) as last_activity_date,
                try_cast(json_extract_string(payload_json, '$.course_pass') as integer) as course_pass,
                json_extract_string(payload_json, '$.final_grade') as final_grade,
                try_cast(json_extract_string(payload_json, '$.has_certificate') as integer) as has_certificate,
                try_cast(json_extract_string(payload_json, '$.cert_date') as timestamp) as cert_date,
                json_extract_string(payload_json, '$.learning_status') as learning_status
            from unified_records
            where source_path = 'parquet/fact_student_course.parquet'
              and payload_json is not null
            qualify row_number() over (
                partition by
                    try_cast(json_extract_string(payload_json, '$.user_id') as bigint),
                    json_extract_string(payload_json, '$.course_id')
                order by try_cast(json_extract_string(payload_json, '$.last_activity_date') as timestamp) desc nulls last
            ) = 1
            """
        )

        log("creating dashboard_agent_course_dim")
        con.execute(
            """
            create or replace table dashboard_agent_course_dim as
            select
                course_id,
                any_value(subject_name) as subject_name,
                any_value(department_name) as department_name,
                any_value(course_type) as course_type,
                any_value(course_org_name) as course_org_name,
                any_value(course_faculty_name) as course_faculty_name,
                any_value(course_teacher_name) as course_teacher_name,
                count(*) as user_course_rows,
                count(distinct user_id) as enrolled_users,
                sum(coalesce(course_activity_count, 0)) as total_course_activity_count,
                avg(avg_module_grade) as avg_module_grade,
                max(last_activity_date) as last_activity_date
            from dashboard_agent_user_course_fact
            where course_id is not null and course_id <> ''
            group by course_id
            """
        )

        log("creating dashboard_agent_anonymous_user_map")
        con.execute(
            """
            create or replace table dashboard_agent_anonymous_user_map as
            select
                json_extract_string(payload_json, '$.anonymous_user_id') as anonymous_user_id,
                try_cast(json_extract_string(payload_json, '$.user_id') as bigint) as user_id,
                nullif(json_extract_string(payload_json, '$.course_id'), '') as course_id
            from unified_records
            where source_path in (
                'edx-mysql/student_anonymoususerid/student_anonymoususerid.json',
                'edx-mysql/student_anonymoususerid.json'
            )
              and payload_json is not null
              and nullif(json_extract_string(payload_json, '$.anonymous_user_id'), '') is not null
            qualify row_number() over (
                partition by
                    json_extract_string(payload_json, '$.anonymous_user_id'),
                    nullif(json_extract_string(payload_json, '$.course_id'), '')
                order by record_index
            ) = 1
            """
        )

        log("creating dashboard_agent_activity_joined")
        con.execute(
            """
            create or replace table dashboard_agent_activity_joined as
            with activity as (
                select
                    record_index as activity_record_index,
                    try_cast(replace(substr(json_extract_string(payload_json, '$."@timestamp"'), 1, 19), 'T', ' ') as timestamp) as event_at,
                    json_extract_string(payload_json, '$.timestamp') as raw_timestamp,
                    nullif(json_extract_string(payload_json, '$.userID'), '') as activity_user_id,
                    nullif(json_extract_string(payload_json, '$.courseID'), '') as course_id,
                    nullif(json_extract_string(payload_json, '$.sessionID'), '') as session_id,
                    nullif(json_extract_string(payload_json, '$.flowID'), '') as flow_id,
                    nullif(json_extract_string(payload_json, '$.appID'), '') as app_id,
                    nullif(json_extract_string(payload_json, '$.eventCategory'), '') as event_category,
                    nullif(json_extract_string(payload_json, '$.event'), '') as event_name,
                    payload_json as activity_payload_json
                from unified_records
                where source_path = 'edx-elastic/ae-activity-data-stream.json'
                  and payload_json is not null
            ),
            mapped as (
                select
                    a.*,
                    coalesce(map_course.user_id, map_any.user_id, try_cast(a.activity_user_id as bigint)) as mapped_user_id
                from activity a
                left join dashboard_agent_anonymous_user_map map_course
                  on map_course.anonymous_user_id = a.activity_user_id
                 and map_course.course_id = a.course_id
                left join dashboard_agent_anonymous_user_map map_any
                  on map_any.anonymous_user_id = a.activity_user_id
                 and map_any.course_id is null
            )
            select
                m.activity_record_index,
                m.event_at,
                date_trunc('hour', m.event_at) as event_hour,
                cast(m.event_at as date) as event_date,
                m.raw_timestamp,
                m.activity_user_id,
                m.mapped_user_id as user_id,
                coalesce(uc.username, u.username) as username,
                coalesce(uc.email, u.email) as email,
                coalesce(uc.full_name, u.full_name) as full_name,
                coalesce(uc.school_name, u.school_name) as school_name,
                coalesce(uc.school_province, u.school_province) as school_province,
                coalesce(uc.province, u.school_province) as province,
                u.institute_id,
                u.level_of_education,
                m.course_id,
                c.subject_name,
                c.department_name,
                c.course_type,
                c.course_org_name,
                c.course_faculty_name,
                c.course_teacher_name,
                c.enrolled_users,
                c.total_course_activity_count,
                c.avg_module_grade as course_avg_module_grade,
                c.last_activity_date as course_last_activity_date,
                uc.enroll_date,
                uc.course_activity_count as user_course_activity_count,
                uc.avg_module_grade as user_course_avg_module_grade,
                uc.max_module_grade as user_course_max_module_grade,
                uc.course_pass,
                uc.final_grade,
                uc.has_certificate,
                uc.cert_date,
                uc.learning_status,
                m.session_id,
                m.flow_id,
                m.app_id,
                m.event_category,
                m.event_name,
                m.activity_payload_json
            from mapped m
            left join dashboard_agent_user_dim u
              on u.user_id = m.mapped_user_id
            left join dashboard_agent_user_course_fact uc
              on uc.user_id = m.mapped_user_id
             and uc.course_id = m.course_id
            left join dashboard_agent_course_dim c
              on c.course_id = m.course_id
            """
        )

        log("creating dashboard_agent_activity_hourly_cache")
        con.execute(
            """
            create or replace table dashboard_agent_activity_hourly_cache as
            with hourly as (
                select
                    event_hour,
                    count(*) as records,
                    count(distinct activity_user_id) as users
                from dashboard_agent_activity_joined
                where event_hour is not null
                group by event_hour
            ),
            first_seen as (
                select
                    activity_user_id,
                    min(event_hour) as first_event_hour
                from dashboard_agent_activity_joined
                where event_hour is not null
                  and activity_user_id is not null
                  and activity_user_id <> ''
                group by activity_user_id
            ),
            first_seen_counts as (
                select
                    first_event_hour as event_hour,
                    count(*) as first_seen_users
                from first_seen
                group by first_event_hour
            )
            select
                'edx-elastic/ae-activity-data-stream.json' as source_path,
                '@timestamp' as timestamp_field,
                'userID' as user_field,
                strftime(h.event_hour, '%Y-%m-%d %H:%M') as label,
                h.records,
                h.users,
                sum(h.records) over (
                    order by h.event_hour
                    rows between unbounded preceding and current row
                ) as cumulative_records,
                sum(coalesce(f.first_seen_users, 0)) over (
                    order by h.event_hour
                    rows between unbounded preceding and current row
                ) as cumulative_users,
                current_timestamp as updated_at
            from hourly h
            left join first_seen_counts f using (event_hour)
            """
        )

        con.execute("checkpoint")
        for table in (
            "dashboard_agent_user_dim",
            "dashboard_agent_user_course_fact",
            "dashboard_agent_course_dim",
            "dashboard_agent_anonymous_user_map",
            "dashboard_agent_activity_joined",
            "dashboard_agent_activity_hourly_cache",
        ):
            count = con.execute(f"select count(*) from {table}").fetchone()[0]
            log(f"{table}: {count:,} rows")
    finally:
        con.close()
    log(f"done in {time.time() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build joined analytical DuckDB mart tables.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    build_joined_mart(args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
