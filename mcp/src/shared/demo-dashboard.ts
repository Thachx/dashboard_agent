import type { DashboardDocument } from "./dashboard-contract.js";

export const demoDashboard: DashboardDocument = {
  schemaVersion: "1.0",
  dashboardId: "demo-users-by-learning-status",
  query: "Tell me about users",
  title: "Users by Learning Status",
  subtitle: "Dashboard generated from a validated multi-source analytical plan.",
  locale: "en",
  timezone: "Asia/Bangkok",
  generatedAt: "2026-07-23T00:00:00.000Z",
  summary: "103.3K users are represented across three learning statuses and six departments.",
  metrics: [
    { id: "distinctUsers", label: "Users", value: 103300, hint: "Distinct values included in the aggregate" },
    { id: "distinctDimensionValues", label: "Learning statuses", value: 3, hint: "Groups included in the comparison" },
  ],
  charts: [
    {
      id: "learning-status",
      title: "Users by Learning Status",
      chartType: "donut",
      field: "learning_status",
      data: [
        { label: "In Progress", value: 68200 },
        { label: "Passed", value: 48700 },
        { label: "Inactive", value: 5900 },
      ],
    },
    {
      id: "department",
      title: "Users by Department",
      chartType: "column",
      field: "department_name",
      data: [
        { label: "Nectec", value: 97900 },
        { label: "Adtep", value: 11300 },
        { label: "BDI", value: 2500 },
        { label: "ETDA", value: 1200 },
        { label: "IPST", value: 963 },
        { label: "BU", value: 548 },
      ],
    },
    {
      id: "teacher",
      title: "Users by Course Teacher",
      chartType: "treemap",
      field: "course_teacher_name",
      data: [
        { label: "Nectec|ipst", value: 51900 },
        { label: "Ipst|nectec", value: 49200 },
        { label: "Teacher group A", value: 8300 },
        { label: "Learning center", value: 2500 },
        { label: "Instructor B", value: 708 },
      ],
    },
  ],
  layout: [
    { type: "metric", id: "distinctUsers" },
    { type: "metric", id: "distinctDimensionValues" },
    { type: "chart", slotId: "learning-status" },
    { type: "chart", slotId: "department" },
    { type: "chart", slotId: "teacher", span: 2 },
  ],
  sources: [
    { id: "fact_student_course", label: "Fact Student Course", path: "parquet/fact_student_course.parquet", objectType: "parquet" },
    { id: "dim_user", label: "Dim User", path: "raw-parquet/dim_user.parquet", objectType: "parquet" },
  ],
  reasoningSummary: [
    { stage: "Interpret request", detail: "Selected a broad user overview with status and organization comparisons." },
    { stage: "Resolve data model", detail: "Combined graph lineage with aggregate data from DuckDB." },
    { stage: "Choose layout", detail: "Used compact composition, comparison, and hierarchy charts based on field shape." },
  ],
  execution: { source: "fixture", graphUsed: true, duckdbUsed: true },
};

