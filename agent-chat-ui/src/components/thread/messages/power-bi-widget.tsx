"use client";

import { useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  ChevronDown,
  Clock,
  Database,
  ListChecks,
  MousePointerClick,
  X,
  Users,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Funnel,
  FunnelChart,
  LabelList,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  Treemap,
  XAxis,
  YAxis,
} from "recharts";

type ChartDatum = {
  label: string;
  series?: string;
  value?: number;
  score?: number;
};

type SourceDisclosureItem = {
  label: string;
  value?: unknown;
  samples?: unknown[];
};

type SourceModalState = {
  title: string;
  samples: unknown[];
} | null;

type DecisionTraceItem = {
  step?: string;
  stage?: string;
  detail: string;
  evidence?: string[];
};

const CHART_COLORS = ["#2563eb", "#10b981", "#f97316", "#7c3aed", "#ec4899", "#64748b"];
const HUMAN_LABEL_RE = /([a-z0-9])([A-Z])/g;
const HUMAN_KEEP_ALL_CAPS = new Set(["API", "CSV", "DB", "ETAG", "ID", "JSON", "LLM", "SQL", "S3", "UI", "URL", "UTC"]);
const THAI_SOURCE_LABELS: Record<string, string> = {
  "Dashboard sources": "แหล่งข้อมูลแดชบอร์ด",
  "Original source": "แหล่งข้อมูลต้นทาง",
  "Derived table": "ตารางที่ประมวลผลแล้ว",
  "Object type": "ชนิดข้อมูล",
  "Rows used": "จำนวนแถวที่ใช้",
  Coverage: "ขอบเขตข้อมูล",
  Fields: "ฟิลด์ข้อมูล",
  "Query plan": "แผนการสืบค้น",
  Source: "แหล่งข้อมูล",
  Updated: "อัปเดตล่าสุด",
};

type DatasetSummary = {
  key: string;
  label?: string;
  source_paths?: string[];
  sourcePaths?: string[];
  s3Uri?: string;
  bucket?: string;
  objectType?: string;
  sizeBytes?: number | string;
  ETAG?: string;
  lastModified?: string;
  score?: number;
  matchedFields?: string[];
};

type ActivityRecord = {
  source?: string;
  index?: number;
  timestamp?: string;
  "@timestamp"?: string;
  appID?: string;
  eventCategory?: string;
  event?: string;
  userID?: string;
  courseID?: string;
  [key: string]: unknown;
};

type ActivityDatasetMeta = {
  source?: string;
  source_paths?: string[];
  source_samples?: Record<string, unknown[]>;
  s3_uri?: string;
  bucket?: string;
  key?: string;
  object_type?: string;
  size_bytes?: number | string;
  last_modified?: string;
  sample_record_count?: number;
  content_sample_ranges?: number;
  content_sample_bytes?: number | string;
  sample_fields?: unknown;
  totalRecords?: number | string;
  sampleRecords?: number;
  isFullAggregate?: boolean;
  distinctEvents?: number;
  distinctCourses?: number;
  distinctUsers?: number;
};

type ChartSlot = {
  id: string;
  title: string;
  chartType: string;
  field: string;
  splitField?: string;
  reason?: string;
  data: ChartDatum[];
};

type LayoutBlock =
  | { type: "metric"; id: string; span?: number }
  | { type: "chart"; slotId: string; span?: number }
  | { type: "source"; span?: number }
  | { type: "records"; span?: number };

type ActivityDashboardPayload = {
  datasets?: Record<string, ActivityDatasetMeta> | ActivityDatasetMeta[];
  summary?: {
    source?: string;
    sampleRecords?: number;
    totalRecords?: number | string;
    isFullAggregate?: boolean;
    distinctEvents?: number;
    distinctCourses?: number;
    distinctUsers?: number;
    sourcePaths?: string[];
    sourceSamples?: Record<string, unknown[]>;
    presentationLanguage?: string;
    metricLabels?: Record<string, string>;
  };
  charts?: {
    events?: ChartDatum[];
    categories?: ChartDatum[];
    courses?: ChartDatum[];
    apps?: ChartDatum[];
    [key: string]: ChartDatum[] | undefined;
  };
  chartPlan?: Array<{
    id: string;
    title: string;
    chartType: string;
    field: string;
    reason?: string;
  }>;
  chartSlots?: ChartSlot[];
  layoutSpec?: {
    title?: string;
    subtitle?: string;
    blocks?: LayoutBlock[];
  };
  decisionTrace?: DecisionTraceItem[];
  records?: ActivityRecord[];
};

export type GraphDashboardWidgetPayload = {
  version: number;
  kind: "graph-dashboard-widget";
  title: string;
  status: {
    objects?: number;
    nodes?: number;
    edges?: number;
    updatedAt?: string | number | null;
  };
  results?: Array<{
    id?: string;
    label?: string;
    path?: string;
    source?: string;
    value?: unknown;
    text?: string;
    score?: number;
  }>;
  datasets?: DatasetSummary[];
  activity?: ActivityDashboardPayload;
  charts?: {
    status?: ChartDatum[];
    results?: ChartDatum[];
  };
};

function formatValue(value: unknown, fallback?: string): string {
  if (value === null || value === undefined || value === "") {
    return fallback ?? "";
  }
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function previewText(value: unknown, fallback?: string, maxLength = 420): string {
  const text = formatValue(value, fallback).replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
}

function sourceValues(value: unknown): string[] {
  if (value === null || value === undefined || value === "") return [];
  if (Array.isArray(value)) return value.map((item) => formatValue(item)).filter(Boolean);
  return [formatValue(value)].filter(Boolean);
}

function uniqueValues(values: Array<string | undefined | null>): string[] {
  return Array.from(new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value))));
}

function isInternalDashboardSource(value: string): boolean {
  const normalized = value.toLowerCase().replace(/\\/g, "/");
  return (
    normalized.includes("dashboard_agent_") ||
    normalized.includes("dashboard agent ") ||
    normalized.startsWith("duckdb/") ||
    normalized.startsWith("graph/")
  );
}

function publicSourceValues(values: Array<string | undefined | null>): string[] {
  return uniqueValues(values).filter((value) => !isInternalDashboardSource(value));
}

function sampleTitle(value: string): string {
  return humanLabel(value) || value;
}

function sampleRowsForSource(samples: unknown[] | undefined, source: string): unknown[] {
  const rows = samples ?? [];
  if (!rows.length) return [];
  const sourceLower = source.toLowerCase();
  const directMatches = rows.filter((row) => {
    if (!row || typeof row !== "object") return false;
    const record = row as Record<string, unknown>;
    const rowSource = String(record.source ?? record.source_path ?? record.sourceFile ?? record.path ?? "").toLowerCase();
    return rowSource.includes(sourceLower) || sourceLower.includes(rowSource);
  });
  return directMatches.slice(0, 50);
}

function sampleObjectRows(samples: unknown[]): Record<string, unknown>[] {
  return samples.filter((sample): sample is Record<string, unknown> => {
    return Boolean(sample) && typeof sample === "object" && !Array.isArray(sample);
  });
}

function sampleTableColumns(rows: Record<string, unknown>[]): string[] {
  const columns: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!columns.includes(key)) columns.push(key);
      if (columns.length >= 12) return columns;
    }
  }
  return columns;
}

function SampleDataModal({
  state,
  onClose,
  language = "en",
}: {
  state: SourceModalState;
  onClose: () => void;
  language?: string;
}) {
  const [viewMode, setViewMode] = useState<"table" | "raw">("table");
  if (!state) return null;
  const tableRows = sampleObjectRows(state.samples);
  const tableColumns = sampleTableColumns(tableRows);
  const canShowTable = tableRows.length > 0 && tableColumns.length > 0;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-3 sm:p-5" role="dialog" aria-modal="true">
      <div className="flex h-[90vh] max-h-[92vh] w-full max-w-[min(96vw,1280px)] flex-col overflow-hidden rounded-lg border bg-background shadow-xl">
        <div className="flex items-start justify-between gap-3 border-b px-4 py-3">
          <div className="min-w-0">
            <p className="text-sm font-semibold">{language === "th" ? "ข้อมูลตัวอย่าง" : "Sample data"}</p>
            <p className="mt-1 break-words text-xs text-muted-foreground">{state.title}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-foreground"
            aria-label={language === "th" ? "ปิดข้อมูลตัวอย่าง" : "Close sample data"}
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
        <div className="flex items-center gap-2 border-b px-4 py-2 text-xs">
          <button
            type="button"
            onClick={() => setViewMode("table")}
            disabled={!canShowTable}
            className={`rounded border px-2 py-1 transition ${
              viewMode === "table" ? "bg-foreground text-background" : "bg-background text-foreground/70 hover:text-foreground"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {language === "th" ? "ตาราง" : "Table"}
          </button>
          <button
            type="button"
            onClick={() => setViewMode("raw")}
            className={`rounded border px-2 py-1 transition ${
              viewMode === "raw" ? "bg-foreground text-background" : "bg-background text-foreground/70 hover:text-foreground"
            }`}
          >
            {language === "th" ? "ข้อมูลดิบ" : "Raw"}
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-4">
          {state.samples.length && viewMode === "table" && canShowTable ? (
            <div className="overflow-auto rounded-lg border">
              <table className="min-w-full text-left text-xs">
                <thead className="bg-muted/40 text-muted-foreground">
                  <tr>
                    {tableColumns.map((column) => (
                      <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                        {humanLabel(column)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="border-t">
                      {tableColumns.map((column) => (
                        <td key={column} className="max-w-[360px] break-words px-3 py-2 align-top">
                          {previewText(row[column], undefined, 220)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : state.samples.length ? (
            <div className="space-y-3">
              {state.samples.map((sample, index) => (
                <pre
                  key={index}
                  className="overflow-auto rounded-lg border bg-muted/20 p-3 text-xs leading-relaxed text-foreground/85"
                >
                  {previewText(sample, undefined, 1400)}
                </pre>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              {language === "th" ? "ไม่มีข้อมูลตัวอย่างสำหรับผลลัพธ์นี้" : "No sample data is available in this widget payload."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceDisclosure({
  label,
  items,
  language = "en",
}: {
  label?: string;
  items: SourceDisclosureItem[];
  language?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sampleModal, setSampleModal] = useState<SourceModalState>(null);
  const visibleItems = items
    .map((item) => ({ ...item, value: sourceValues(item.value) }))
    .filter((item) => item.value.length);
  if (!visibleItems.length) return null;

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        className="group inline-flex max-w-full items-center gap-1 text-left text-xs text-foreground/35 transition hover:text-foreground/70"
        aria-expanded={isOpen}
      >
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 transition-transform ${isOpen ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
        <span className="truncate">{label ?? (language === "th" ? "แสดงแหล่งข้อมูล" : "Show source")}</span>
      </button>
      {isOpen ? (
        <dl className="mt-2 grid gap-1 rounded-lg border bg-muted/10 p-3 text-xs">
          {visibleItems.map((item, index) => (
            <div key={`${item.label}-${index}`} className="grid gap-1 sm:grid-cols-[140px_1fr]">
              <dt className="text-muted-foreground">
                {language === "th" ? THAI_SOURCE_LABELS[item.label] ?? item.label : item.label}
              </dt>
              <dd className="flex min-w-0 flex-wrap gap-1.5 font-medium text-foreground/80">
                {item.value.map((value) => {
                  const samples = sampleRowsForSource(item.samples, value);
                  if (!samples.length) {
                    return <span key={value} className="break-words">{value}</span>;
                  }
                  return (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setSampleModal({ title: value, samples })}
                      className="max-w-full rounded border bg-background px-2 py-0.5 text-left text-foreground/70 transition hover:border-foreground/30 hover:text-foreground"
                      title={language === "th" ? `แสดงข้อมูลตัวอย่างของ ${value}` : `Show sample data for ${value}`}
                    >
                      <span className="block truncate">{sampleTitle(value)}</span>
                    </button>
                  );
                })}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      <SampleDataModal state={sampleModal} onClose={() => setSampleModal(null)} language={language} />
    </div>
  );
}

function ReasoningDisclosure({
  trace,
  samples,
  language = "en",
}: {
  trace?: DecisionTraceItem[];
  samples?: unknown[];
  language?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sampleModal, setSampleModal] = useState<SourceModalState>(null);
  const items = (trace ?? []).filter((item) => item.stage || item.step || item.detail);
  if (!items.length) return null;

  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        className="group inline-flex max-w-full items-center gap-1 text-left text-xs text-foreground/35 transition hover:text-foreground/70"
        aria-expanded={isOpen}
      >
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 transition-transform ${isOpen ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
        <span className="truncate">{language === "th" ? "แสดงสรุปการตัดสินใจ" : "Show reasoning"}</span>
      </button>
      {isOpen ? (
        <ol className="mt-2 grid gap-2 rounded-lg border bg-muted/10 p-3 text-xs">
          {items.map((item, index) => (
            <li key={`${item.stage ?? item.step ?? "reasoning"}-${index}`} className="grid gap-1 sm:grid-cols-[140px_1fr]">
              <span className="font-medium text-foreground/75">{humanLabel(item.stage ?? item.step ?? "")}</span>
              <span className="min-w-0 text-foreground/80">
                <span className="block break-words">{item.detail}</span>
                {item.evidence?.length ? (
                  <span className="mt-1 flex flex-wrap gap-1.5">
                    {item.evidence.slice(0, 8).map((entry) => {
                      const sourceSamples = sampleRowsForSource(samples, entry);
                      if (!sourceSamples.length) {
                        return (
                          <span key={entry} className="max-w-full rounded border bg-background px-1.5 py-0.5 text-foreground/60">
                            {entry}
                          </span>
                        );
                      }
                      return (
                        <button
                          key={entry}
                          type="button"
                          onClick={() => setSampleModal({ title: entry, samples: sourceSamples })}
                          className="max-w-full rounded border bg-background px-1.5 py-0.5 text-left text-foreground/70 transition hover:border-foreground/30 hover:text-foreground"
                          title={language === "th" ? `แสดงข้อมูลตัวอย่างของ ${entry}` : `Show sample data for ${entry}`}
                        >
                          <span className="block truncate">{entry}</span>
                        </button>
                      );
                    })}
                  </span>
                ) : null}
              </span>
            </li>
          ))}
        </ol>
      ) : null}
      <SampleDataModal state={sampleModal} onClose={() => setSampleModal(null)} language={language} />
    </div>
  );
}

function activitySourceItems(activity?: ActivityDashboardPayload): SourceDisclosureItem[] {
  if (!activity) return [];
  const datasetValues = Array.isArray(activity.datasets)
    ? activity.datasets
    : Object.values(activity.datasets ?? {});
  const sourceMeta = datasetValues[0];
  const summary = activity.summary ?? {};
  const originalSources = publicSourceValues([
    ...(summary.sourcePaths ?? []),
    ...datasetValues.flatMap((dataset) => dataset.source_paths ?? []),
    ...datasetValues.map((dataset) => dataset.s3_uri ?? dataset.source ?? "").filter(Boolean),
  ]);
  const sourceSamples = {
    ...(summary.sourceSamples ?? {}),
    ...datasetValues.reduce<Record<string, unknown[]>>((samples, dataset) => {
      return { ...samples, ...(dataset.source_samples ?? {}) };
    }, {}),
  };
  const chartPlan = activity.chartPlan ?? activity.chartSlots ?? [];
  const sourceSampleRows = Object.entries(sourceSamples).flatMap(([source, rows]) =>
    (rows ?? []).map((row) => (row && typeof row === "object" ? { ...(row as Record<string, unknown>), source } : { source, value: row })),
  );
  const sampleRows = sourceSampleRows.length
    ? sourceSampleRows
    : activity.records?.length
      ? activity.records
      : Object.entries(activity.charts ?? {}).flatMap(([chart, rows]) =>
          (rows ?? []).slice(0, 8).map((row) => ({ chart, ...row })),
        );
  return [
    {
      label: "Dashboard sources",
      value: originalSources,
      samples: sampleRows,
    },
    { label: "Object type", value: sourceMeta?.object_type },
    { label: "Rows used", value: summary.totalRecords ?? summary.sampleRecords ?? sourceMeta?.totalRecords },
    { label: "Coverage", value: summary.isFullAggregate ? "Full aggregate" : "Sample records" },
    { label: "Fields", value: chartPlan.map((slot) => humanLabel(slot.field)).filter(Boolean) },
    { label: "Query plan", value: chartPlan.map((slot) => slot.reason).filter(Boolean).join(" ") },
  ];
}

function activitySourceSampleRows(activity?: ActivityDashboardPayload): unknown[] {
  if (!activity) return [];
  const datasetValues = Array.isArray(activity.datasets)
    ? activity.datasets
    : Object.values(activity.datasets ?? {});
  const summary = activity.summary ?? {};
  const sourceSamples = {
    ...(summary.sourceSamples ?? {}),
    ...datasetValues.reduce<Record<string, unknown[]>>((samples, dataset) => {
      return { ...samples, ...(dataset.source_samples ?? {}) };
    }, {}),
  };
  return Object.entries(sourceSamples).flatMap(([source, rows]) =>
    (rows ?? []).map((row) => (row && typeof row === "object" ? { ...(row as Record<string, unknown>), source } : { source, value: row })),
  );
}

function graphSourceItems(payload: GraphDashboardWidgetPayload): SourceDisclosureItem[] {
  const datasets = payload.datasets ?? [];
  const results = payload.results ?? [];
  const originalSources = publicSourceValues(
    datasets.flatMap((dataset) => {
      const sourcePaths = dataset.source_paths ?? dataset.sourcePaths ?? [];
      return sourcePaths.length ? sourcePaths : [dataset.s3Uri ?? dataset.key];
    }),
  );
  const evidenceSamples = results.map((item) => ({
    label: item.label,
    source: item.source ?? item.path ?? item.id,
    score: item.score,
    value: item.value,
    text: item.text,
  }));
  return [
    { label: "Graph objects", value: payload.status?.objects },
    { label: "Graph nodes", value: payload.status?.nodes },
    { label: "Graph edges", value: payload.status?.edges },
    {
      label: "Dashboard sources",
      value: originalSources,
      samples: evidenceSamples,
    },
    { label: "Matched fields", value: datasets.flatMap((dataset) => dataset.matchedFields ?? []) },
    { label: "Evidence nodes", value: results.map((item) => item.source ?? item.path ?? item.id) },
  ];
}

function formatNumber(value: unknown): string {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) return String(value ?? "");
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(numeric);
}

function formatBytes(value: unknown): string {
  const bytes = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(bytes)) return formatValue(value, "Unknown");
  return `${Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(bytes)}B`;
}

function compactLabel(value: string): string {
  return value.length > 18 ? `${value.slice(0, 15)}...` : value;
}

function shortId(value: unknown, visible = 8): string {
  const text = formatValue(value, "Unknown");
  if (text.length <= visible * 2 + 3) return text;
  return `${text.slice(0, visible)}...${text.slice(-visible)}`;
}

function courseName(value: unknown): string {
  const text = formatValue(value, "Unknown course");
  if (!text.startsWith("course-v1:")) return shortId(text, 16);
  const raw = text.replace("course-v1:", "");
  const [org, course, run] = raw.split("+");
  return [org, course, run].filter(Boolean).join(" / ");
}

function formatDateTime(value: unknown): string {
  const text = formatValue(value);
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  return Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function humanLabel(value: string, kind?: "course" | "user"): string {
  if (kind === "course") return courseName(value);
  if (kind === "user") return shortId(value, 6);
  const text = String(value ?? "").trim();
  if (!text) return "Unknown";
  const basename = text.replace(/\\/g, "/").split("/").pop() ?? text;
  const stripped = basename
    .replace(/^\$\.?/, "")
    .replace(/\.(json|csv|tsv|parquet|ndjson|jsonl|sql|db|duckdb|txt)$/i, "")
    .replace(HUMAN_LABEL_RE, "$1 $2")
    .replace(/[_\-.]+/g, " ");
  const words = stripped
    .split(/\s+/)
    .map((word) => {
      const trimmed = word.trim();
      if (!trimmed) return "";
      const upper = trimmed.toUpperCase();
      if (HUMAN_KEEP_ALL_CAPS.has(upper)) return upper;
      if (/^[A-Z0-9]{2,4}$/.test(trimmed)) return upper;
      if (/^\d+$/.test(trimmed)) return trimmed;
      return trimmed.slice(0, 1).toUpperCase() + trimmed.slice(1).toLowerCase();
    })
    .filter(Boolean);
  return words.length ? words.join(" ") : "Unknown";
}

function BreakdownList({
  data,
  title,
  kind,
}: {
  data: ChartDatum[];
  title: string;
  kind?: "course" | "user";
}) {
  if (!data.length) return null;
  const max = Math.max(...data.map((item) => Number(item.value ?? item.score ?? 0)), 1);

  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 space-y-3">
        {data.slice(0, 6).map((item) => {
          const value = Number(item.value ?? item.score ?? 0);
          return (
            <div key={`${title}-${item.label}`}>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="min-w-0 truncate font-medium" title={item.label}>
                  {humanLabel(item.label, kind)}
                </span>
                <span className="shrink-0 text-muted-foreground">{formatNumber(value)}</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-blue-600"
                  style={{ width: `${Math.max(6, (value / max) * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
}: {
  label: string;
  value: unknown;
  hint: string;
  icon: typeof Activity;
}) {
  return (
    <div className="min-w-0 rounded-lg border bg-background p-3 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="mt-1 text-2xl font-semibold">{formatNumber(value)}</p>
        </div>
        <div className="rounded-lg bg-blue-50 p-2 text-blue-700">
          <Icon className="h-4 w-4" aria-hidden="true" />
        </div>
      </div>
      <p className="mt-2 truncate text-xs text-muted-foreground" title={hint}>{hint}</p>
    </div>
  );
}

function DonutChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data
    .slice(0, 6)
    .map((item) => ({ name: item.label, value: Number(item.value ?? item.score ?? 0) }))
    .filter((item) => item.value > 0);
  const total = chartData.reduce((sum, item) => sum + item.value, 0);
  if (!chartData.length) return null;

  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 grid min-h-44 gap-3 sm:grid-cols-[160px_1fr]">
        <div className="relative h-44">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                innerRadius={42}
                outerRadius={62}
                paddingAngle={3}
              >
                {chartData.map((item, index) => (
                  <Cell key={item.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => formatNumber(value)} />
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <p className="text-lg font-semibold">{formatNumber(total)}</p>
              <p className="text-[10px] text-muted-foreground">records</p>
            </div>
          </div>
        </div>
        <div className="space-y-2 self-center">
          {chartData.map((item, index) => (
            <div key={item.name} className="flex items-center justify-between gap-2 text-xs">
              <span className="flex min-w-0 items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                />
                <span className="truncate" title={item.name}>{humanLabel(item.name)}</span>
              </span>
              <span className="font-medium">{formatNumber(item.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MiniColumnChart({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.slice(0, 6).map((item) => ({
    label: humanLabel(item.label),
    value: Number(item.value ?? item.score ?? 0),
  }));
  if (!chartData.length) return null;

  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 h-44">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -24 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={0} tickFormatter={compactLabel} />
            <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} fill="#2563eb" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PieChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.slice(0, 8).map((item) => ({
    name: humanLabel(item.label),
    value: Number(item.value ?? item.score ?? 0),
  })).filter((item) => item.value > 0);
  if (!chartData.length) return null;
  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={chartData} dataKey="value" nameKey="name" outerRadius={88} paddingAngle={2}>
              {chartData.map((item, index) => <Cell key={item.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
              <LabelList dataKey="name" position="outside" className="fill-foreground text-[10px]" />
            </Pie>
            <Tooltip formatter={(value) => formatNumber(value)} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function RadarChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.slice(0, 8).map((item) => ({
    label: humanLabel(item.label),
    value: Number(item.value ?? item.score ?? 0),
  }));
  if (chartData.length < 3) return null;
  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={chartData} outerRadius="72%">
            <PolarGrid />
            <PolarAngleAxis dataKey="label" tick={{ fontSize: 10 }} />
            <Radar dataKey="value" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
            <Tooltip formatter={(value) => formatNumber(value)} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function RadialBarChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.slice(0, 6).map((item, index) => ({
    name: humanLabel(item.label),
    value: Number(item.value ?? item.score ?? 0),
    fill: CHART_COLORS[index % CHART_COLORS.length],
  })).filter((item) => item.value > 0);
  if (!chartData.length) return null;
  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart data={chartData} innerRadius="18%" outerRadius="92%" startAngle={90} endAngle={-270}>
            <RadialBar dataKey="value" background cornerRadius={5} />
            <Legend iconSize={8} layout="vertical" verticalAlign="middle" align="right" />
            <Tooltip formatter={(value) => formatNumber(value)} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FunnelChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.slice(0, 8).map((item, index) => ({
    name: humanLabel(item.label),
    value: Number(item.value ?? item.score ?? 0),
    fill: CHART_COLORS[index % CHART_COLORS.length],
  })).filter((item) => item.value > 0).sort((left, right) => right.value - left.value);
  if (chartData.length < 2) return null;
  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <FunnelChart>
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Funnel data={chartData} dataKey="value" nameKey="name" isAnimationActive={false}>
              <LabelList position="right" fill="currentColor" stroke="none" dataKey="name" />
            </Funnel>
          </FunnelChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function seriesChartData(data: ChartDatum[]) {
  const series = Array.from(new Set(data.map((item) => String(item.series ?? "")).filter(Boolean))).slice(0, 6);
  const rows = new Map<string, Record<string, string | number>>();
  data.forEach((item) => {
    const label = String(item.label ?? "");
    const seriesName = String(item.series ?? "");
    if (!label || !seriesName || !series.includes(seriesName)) return;
    const row = rows.get(label) ?? { label };
    row[seriesName] = Number(row[seriesName] ?? 0) + Number(item.value ?? item.score ?? 0);
    rows.set(label, row);
  });
  return { series, rows: Array.from(rows.values()) };
}

function MultiSeriesLineChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chart = seriesChartData(data);
  if (!chart.rows.length || !chart.series.length) return null;
  return (
    <div className="min-w-0 rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chart.rows} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={24} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={formatNumber} width={44} />
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Legend iconSize={8} />
            {chart.series.map((seriesName, index) => (
              <Line key={seriesName} type="monotone" dataKey={seriesName} stroke={CHART_COLORS[index % CHART_COLORS.length]} strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StackedColumnChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chart = seriesChartData(data);
  if (!chart.rows.length || !chart.series.length) return null;
  return (
    <div className="min-w-0 rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chart.rows} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} tickFormatter={compactLabel} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={formatNumber} width={44} />
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Legend iconSize={8} />
            {chart.series.map((seriesName, index) => (
              <Bar key={seriesName} dataKey={seriesName} stackId="total" fill={CHART_COLORS[index % CHART_COLORS.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StackedBarChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const labels = Array.from(new Set(data.map((item) => String(item.label ?? "")).filter(Boolean))).slice(0, 8);
  const series = Array.from(new Set(data.map((item) => String(item.series ?? "")).filter(Boolean))).slice(0, 6);
  const rows = labels.map((label) => {
    const segments = series.map((seriesName, index) => {
      const value = data
        .filter((item) => String(item.label) === label && String(item.series) === seriesName)
        .reduce((sum, item) => sum + Number(item.value ?? item.score ?? 0), 0);
      return {
        series: seriesName,
        value,
        color: CHART_COLORS[index % CHART_COLORS.length],
      };
    }).filter((segment) => segment.value > 0);
    return {
      label,
      displayLabel: humanLabel(label, "course"),
      total: segments.reduce((sum, segment) => sum + segment.value, 0),
      segments,
    };
  });
  const max = Math.max(...rows.map((row) => row.total), 1);
  if (!rows.length || !series.length) return null;

  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 space-y-3">
        {rows.map((row) => (
          <div key={row.label} className="min-w-0">
            <div className="mb-1 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 text-xs">
              <span className="block min-w-0 truncate font-medium leading-4" title={row.label}>
                {row.displayLabel}
              </span>
              <span className="whitespace-nowrap text-right text-muted-foreground">{formatNumber(row.total)}</span>
            </div>
            <div className="flex h-3 overflow-hidden rounded-full bg-muted">
              {row.segments.map((segment, index) => (
                <div
                  key={`${row.label}-${segment.series}`}
                  className="h-full"
                  title={`${humanLabel(segment.series)}: ${formatNumber(segment.value)}`}
                  style={{
                    width: `${Math.max(3, (segment.value / max) * 100)}%`,
                    backgroundColor: segment.color,
                    borderTopLeftRadius: index === 0 ? 999 : 0,
                    borderBottomLeftRadius: index === 0 ? 999 : 0,
                    borderTopRightRadius: index === row.segments.length - 1 ? 999 : 0,
                    borderBottomRightRadius: index === row.segments.length - 1 ? 999 : 0,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {series.map((seriesName, index) => (
          <span key={seriesName} className="inline-flex items-center gap-1">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
            />
            {humanLabel(seriesName)}
          </span>
        ))}
      </div>
    </div>
  );
}

function TimelineChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.map((item) => ({
    label: item.label,
    value: Number(item.value ?? item.score ?? 0),
  }));
  if (!chartData.length) return null;

  return (
    <div className="min-w-0 rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={24} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={formatNumber} allowDecimals={false} width={42} />
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#2563eb"
              strokeWidth={3}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function AreaChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data.map((item) => ({
    label: item.label,
    value: Number(item.value ?? item.score ?? 0),
  }));
  if (!chartData.length) return null;

  return (
    <div className="min-w-0 rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={24} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={formatNumber} allowDecimals={false} width={42} />
            <Tooltip formatter={(value) => formatNumber(value)} />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#2563eb"
              strokeWidth={2}
              fill="#93c5fd"
              fillOpacity={0.55}
              activeDot={{ r: 5 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TreemapChartCard({ data, title }: { data: ChartDatum[]; title: string }) {
  const chartData = data
    .slice(0, 12)
    .map((item) => ({
      name: humanLabel(item.label),
      rawLabel: item.label,
      value: Number(item.value ?? item.score ?? 0),
    }))
    .filter((item) => item.value > 0);
  if (!chartData.length) return null;

  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="mt-3 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <Treemap
            data={chartData}
            dataKey="value"
            nameKey="name"
            stroke="#ffffff"
            fill="#2563eb"
            isAnimationActive={false}
          >
            <Tooltip
              formatter={(value) => formatNumber(value)}
              labelFormatter={(label) => String(label)}
            />
          </Treemap>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StatSlotCard({
  data,
  title,
  field,
}: {
  data: ChartDatum[];
  title: string;
  field: string;
}) {
  const item = data[0];
  if (!item) return null;
  return (
    <div className="rounded-lg border bg-muted/10 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <p className="mt-3 text-2xl font-semibold">{humanLabel(item.label)}</p>
      <p className="mt-1 text-xs text-muted-foreground">
        All {formatNumber(item.value)} sampled records share this {field}.
      </p>
    </div>
  );
}

function ChartSlotCard({
  slot,
}: {
  slot: {
    id: string;
    title: string;
    chartType: string;
    field: string;
    reason?: string;
    data: ChartDatum[];
  };
}) {
  const kind = slot.field === "courseID" ? "course" : slot.field === "userID" ? "user" : undefined;
  if (slot.chartType === "multi_line") {
    return <MultiSeriesLineChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "stacked_column") {
    return <StackedColumnChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.data.some((item) => item.series)) {
    return <StackedBarChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "line") {
    return <TimelineChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "area") {
    return <AreaChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "stat") {
    return <StatSlotCard data={slot.data} title={slot.title} field={slot.field} />;
  }
  if (slot.chartType === "donut") {
    return <DonutChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "pie") {
    return <PieChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "radar") {
    return <RadarChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "radial_bar") {
    return <RadialBarChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "funnel") {
    return <FunnelChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "column") {
    return <MiniColumnChart data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "stacked_bar") {
    return <StackedBarChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "treemap") {
    return <TreemapChartCard data={slot.data} title={slot.title} />;
  }
  return <BreakdownList data={slot.data} title={slot.title} kind={kind} />;
}

function chartUsesFullRow(slot: ChartSlot, requestedSpan?: number) {
  if (requestedSpan !== 2) return false;
  if (slot.data.some((item) => item.series)) return true;
  return ["multi_line", "line", "area", "stacked_column", "stacked_bar", "treemap"].includes(
    slot.chartType,
  );
}

function chartUsesCompactColumn(slot: ChartSlot) {
  return ["donut", "pie", "radar", "radial_bar", "stat"].includes(slot.chartType);
}

function chartGridSpans(
  charts: Array<{ slot: ChartSlot; requestedSpan?: number }>,
) {
  const spans = Array.from({ length: charts.length }, () => 12);
  let index = 0;

  while (index < charts.length) {
    const current = charts[index];
    if (chartUsesFullRow(current.slot, current.requestedSpan)) {
      index += 1;
      continue;
    }

    const next = charts[index + 1];
    if (!next || chartUsesFullRow(next.slot, next.requestedSpan)) {
      index += 1;
      continue;
    }

    const currentCompact = chartUsesCompactColumn(current.slot);
    const nextCompact = chartUsesCompactColumn(next.slot);
    spans[index] = currentCompact === nextCompact ? 6 : currentCompact ? 5 : 7;
    spans[index + 1] = currentCompact === nextCompact ? 6 : nextCompact ? 5 : 7;
    index += 2;
  }

  return spans;
}

function chartSpanClass(span: number) {
  if (span === 5) return "xl:col-span-5";
  if (span === 6) return "xl:col-span-6";
  if (span === 7) return "xl:col-span-7";
  return "xl:col-span-12";
}

function DashboardBarChart({
  data,
  dataKey,
  label,
}: {
  data: ChartDatum[];
  dataKey: "value" | "score";
  label: string;
}) {
  if (!data.length) return null;

  return (
    <div className="h-48 rounded-lg border bg-muted/10 p-3">
      <p className="mb-2 text-xs font-semibold text-muted-foreground">
        {label}
      </p>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tickFormatter={compactLabel}
            tick={{ fontSize: 11 }}
            interval={0}
          />
          <YAxis tick={{ fontSize: 11 }} tickFormatter={formatNumber} width={44} />
          <Tooltip formatter={(value) => formatNumber(value)} />
          <Bar dataKey={dataKey} fill="#2563eb" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function PromptDashboardSection({ activity }: { activity?: ActivityDashboardPayload }) {
  const layoutSpec = activity?.layoutSpec;
  if (!layoutSpec?.blocks?.length) {
    return <ActivityDashboardSection activity={activity} />;
  }

  const sourceItems = activitySourceItems(activity);
  const sourceSampleRows = activitySourceSampleRows(activity);
  const records = activity?.records ?? [];
  const charts = activity?.charts ?? {};
  const chartSlots =
    activity?.chartSlots ??
    Object.entries(charts)
      .filter(([, data]) => Array.isArray(data))
      .map(([id, data]) => ({
        id,
        title: id,
        chartType: id === "timeline" || id === "activityTimeline" ? "line" : "bar",
        field: id,
        data: data as ChartDatum[],
      }));
  const slotById = new Map(chartSlots.map((slot) => [slot.id, slot]));
  const datasetValues = Array.isArray(activity?.datasets)
    ? activity.datasets
    : Object.values(activity?.datasets ?? {});
  const sourceMeta = datasetValues[0];
  const summary = activity?.summary ?? {};
  const language = summary.presentationLanguage === "th" ? "th" : "en";
  const isThai = language === "th";
  const metricLabels =
    ((summary as Record<string, unknown>).metricLabels as Record<string, string> | undefined) ?? {};
  const metricDefs: Record<
    string,
    { label: string; value: number | string | undefined; hint: string; icon: typeof Activity }
  > = {
    totalRecords: {
      label: metricLabels.totalRecords ?? (isThai ? "จำนวนรายการกิจกรรม" : "Activity rows"),
      value: summary.totalRecords ?? sourceMeta?.totalRecords ?? summary.sampleRecords ?? records.length,
      hint: summary.isFullAggregate
        ? (isThai ? "ข้อมูลรวมทั้งหมด" : "Full aggregate")
        : (isThai ? "ตัวอย่างที่ตรงกัน" : "Matched sample"),
      icon: Activity,
    },
    sampleRecords: {
      label: metricLabels.sampleRecords ?? (isThai ? "จำนวนแถวตัวอย่าง" : "Sample rows"),
      value: summary.sampleRecords ?? records.length,
      hint: isThai ? "แถวข้อมูลที่แสดงในแดชบอร์ด" : "Rows available in the widget",
      icon: Database,
    },
    distinctEvents: {
      label: metricLabels.distinctEvents ?? (isThai ? "ประเภทกิจกรรม" : "Event types"),
      value: summary.distinctEvents ?? charts.events?.length,
      hint: isThai ? "จำนวนประเภทกิจกรรมที่ไม่ซ้ำ" : "Distinct event values",
      icon: BarChart3,
    },
    distinctCourses: {
      label: metricLabels.distinctCourses ?? (isThai ? "หลักสูตร" : "Courses"),
      value: summary.distinctCourses ?? charts.courses?.length,
      hint: isThai ? "จำนวนหลักสูตรที่ไม่ซ้ำ" : "Distinct course values",
      icon: BookOpen,
    },
    distinctUsers: {
      label: metricLabels.distinctUsers ?? (isThai ? "ผู้ใช้งาน" : "Users"),
      value: summary.distinctUsers,
      hint: isThai ? "จำนวนผู้ใช้งานที่ไม่ซ้ำ" : "Distinct user values",
      icon: Users,
    },
    distinctDimensionValues: {
      label:
        metricLabels.distinctDimensionValues ??
        (isThai
          ? `จำนวน${formatValue((summary as Record<string, unknown>).topDimensionName, "มิติข้อมูล")}ที่ไม่ซ้ำ`
          : `${formatValue((summary as Record<string, unknown>).topDimensionName, "Dimension")} values`),
      value: (summary as Record<string, unknown>).distinctDimensionValues as number | string | undefined,
      hint: isThai ? "จำนวนค่ามิติที่ไม่ซ้ำ" : "Distinct ranked dimension values",
      icon: Database,
    },
    totalDistinctMeasure: {
      label:
        metricLabels.totalDistinctMeasure ??
        (isThai
          ? `${formatValue((summary as Record<string, unknown>).measureName, "ตัวชี้วัด")}ทั้งหมด`
          : `Total ${formatValue((summary as Record<string, unknown>).measureName, "measure")}`),
      value: (summary as Record<string, unknown>).totalDistinctMeasure as number | string | undefined,
      hint: isThai ? "จำนวนค่าที่ไม่ซ้ำในผลรวม" : "Distinct values included in the aggregate",
      icon: Users,
    },
    topDimensionValue: {
      label:
        metricLabels.topDimensionValue ??
        (isThai
          ? `${formatValue((summary as Record<string, unknown>).measureName, "ตัวชี้วัด")}สูงสุด`
          : `Top ${formatValue((summary as Record<string, unknown>).measureName, "measure")}`),
      value: (summary as Record<string, unknown>).topDimensionValue as number | string | undefined,
      hint: formatValue((summary as Record<string, unknown>).topDimensionLabel, isThai ? "ค่าสูงสุด" : "Top value"),
      icon: Users,
    },
  };

  const renderRecords = () => {
    if (!records.length) return null;
    return (
      <div key="records" className="overflow-x-auto rounded-lg bg-muted/10">
        <table className="min-w-[760px] text-left text-xs">
          <thead className="bg-muted/30 text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">{isThai ? "เวลา" : "Time"}</th>
              <th className="px-3 py-2 font-medium">{isThai ? "กิจกรรม" : "Event"}</th>
              <th className="px-3 py-2 font-medium">{isThai ? "หลักสูตร" : "Course"}</th>
              <th className="px-3 py-2 font-medium">{isThai ? "ผู้ใช้งาน" : "User"}</th>
              <th className="px-3 py-2 font-medium">{isThai ? "บริบท" : "Context"}</th>
            </tr>
          </thead>
          <tbody>
            {records.slice(0, 8).map((record, index) => (
              <tr key={`${record.timestamp ?? record["@timestamp"] ?? index}`} className="border-t border-border">
                <td className="px-3 py-2">{formatDateTime(record["@timestamp"] ?? record.timestamp)}</td>
                <td className="px-3 py-2">{String(record.event ?? record.event_type ?? (isThai ? "กิจกรรม" : "Activity"))}</td>
                <td className="px-3 py-2">{courseName(record.courseID)}</td>
                <td className="px-3 py-2">{shortId(record.userID, 8)}</td>
                <td className="max-w-[260px] px-3 py-2">
                  {String(record.eventCategory ?? record.appID ?? record.source_file ?? "").slice(0, 80)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  const metricBlocks = layoutSpec.blocks.filter((block) => block.type === "metric");
  const chartBlocks = layoutSpec.blocks.filter((block) => block.type === "chart");
  const recordBlocks = layoutSpec.blocks.filter((block) => block.type === "records");
  const visibleCharts = chartBlocks.flatMap((block, index) => {
    if (block.type !== "chart") return [];
    const slot = slotById.get(block.slotId);
    return slot?.data?.length ? [{ block, index, slot }] : [];
  });
  const visibleChartSpans = chartGridSpans(
    visibleCharts.map(({ block, slot }) => ({ slot, requestedSpan: block.span })),
  );

  return (
    <section className="overflow-hidden bg-background">
      <div className="border-b border-border p-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-base font-semibold">{layoutSpec.title ?? (isThai ? "แดชบอร์ดข้อมูล" : "Activity dashboard")}</h3>
          {layoutSpec.subtitle ? (
            <p className="max-w-3xl text-sm text-muted-foreground">{layoutSpec.subtitle}</p>
          ) : null}
          <SourceDisclosure items={sourceItems} language={language} />
          <ReasoningDisclosure trace={activity?.decisionTrace} samples={sourceSampleRows} language={language} />
        </div>
      </div>

      <div className="flex flex-col gap-3 p-4">
        <div
          className={
            metricBlocks.length && visibleCharts.length
              ? "grid gap-3 xl:grid-cols-[minmax(220px,280px)_minmax(0,1fr)] xl:items-start"
              : "grid gap-3"
          }
        >
        {metricBlocks.length ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-1">
            {metricBlocks.map((block, index) => {
              if (block.type !== "metric") return null;
              const metric = metricDefs[block.id];
              if (!metric || metric.value == null) return null;
              return (
                <div
                  key={`${block.type}-${block.id}-${index}`}
                  className="min-w-0"
                >
                  <MetricCard
                    label={metric.label}
                    value={metric.value}
                    hint={metric.hint}
                    icon={metric.icon}
                  />
                </div>
              );
            })}
          </div>
        ) : null}

        {visibleCharts.length ? (
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
            {visibleCharts.map(({ block, index, slot }, visibleIndex) => {
              const span = visibleChartSpans[visibleIndex] ?? 12;
              return (
                <div
                  key={`${block.type}-${block.slotId}-${index}`}
                  className={`h-full min-w-0 [&>*]:h-full ${chartSpanClass(span)}`}
                >
                  <ChartSlotCard slot={slot} />
                </div>
              );
            })}
          </div>
        ) : null}
        </div>

        {recordBlocks.map((block, index) =>
          block.type === "records" ? (
            <div key={`${block.type}-${index}`} className="min-w-0">
              {renderRecords()}
            </div>
          ) : null,
        )}
      </div>
    </section>
  );
}

function ActivityDashboardSection({ activity }: { activity?: ActivityDashboardPayload }) {
  const sourceItems = activitySourceItems(activity);
  const sourceSampleRows = activitySourceSampleRows(activity);
  const records = activity?.records ?? [];
  const charts = activity?.charts ?? {};
  const chartSlots =
    activity?.chartSlots ??
    [
      { id: "events", title: "Event mix", chartType: "donut", field: "event", data: charts.events ?? [] },
      {
        id: "courses",
        title: "Courses in the sample",
        chartType: "horizontal_bar",
        field: "courseID",
        data: charts.courses ?? [],
      },
      {
        id: "categories",
        title: "Event categories",
        chartType: "column",
        field: "eventCategory",
        data: charts.categories ?? [],
      },
      { id: "apps", title: "Applications", chartType: "column", field: "appID", data: charts.apps ?? [] },
    ];
  const summary = activity?.summary ?? {};
  const language = summary.presentationLanguage === "th" ? "th" : "en";
  const isThai = language === "th";
  const sourceMeta = Object.values(activity?.datasets ?? {})[0];
  const sourceName = humanLabel(sourceMeta?.key ?? records[0]?.source ?? "Matched activity data");
  const eventLeader = charts.events?.[0];
  const recordTimes = records
    .map((record) => record["@timestamp"] ?? record.timestamp)
    .filter(Boolean)
    .map((value) => new Date(formatValue(value)))
    .filter((date) => !Number.isNaN(date.getTime()))
    .sort((a, b) => a.getTime() - b.getTime());
  const timeRange =
    recordTimes.length > 1
      ? `${formatDateTime(recordTimes[0].toISOString())} - ${formatDateTime(recordTimes[recordTimes.length - 1].toISOString())}`
      : recordTimes.length === 1
        ? formatDateTime(recordTimes[0].toISOString())
        : (isThai ? "ไม่มีตัวอย่างเวลา" : "No timestamp sample");
  if (!records.length && !Object.keys(charts).length) return null;

  const stats = [
    {
      label: summary.isFullAggregate
        ? (isThai ? "รายการกิจกรรมทั้งหมด" : "Total activity rows")
        : (isThai ? "แถวข้อมูลตัวอย่าง" : "Sampled rows"),
      value: summary.totalRecords ?? summary.sampleRecords ?? records.length,
      hint: summary.isFullAggregate
        ? (isThai ? "สแกนข้อมูลครบทั้งหมด" : "Full file scan")
        : (isThai ? "แถวที่อ่านจากข้อมูลกิจกรรม" : "Rows read from the activity stream"),
      icon: ListChecks,
    },
    {
      label: isThai ? "ประเภทกิจกรรม" : "Event types",
      value: summary.distinctEvents ?? charts.events?.length ?? 0,
      hint: isThai ? "กิจกรรมที่ไม่ซ้ำในตัวอย่าง" : "Unique actions in the sample",
      icon: MousePointerClick,
    },
    {
      label: isThai ? "หลักสูตร" : "Courses",
      value: summary.distinctCourses ?? charts.courses?.length ?? 0,
      hint: isThai ? "รหัสหลักสูตรที่พบ" : "Course IDs represented",
      icon: BookOpen,
    },
    {
      label: isThai ? "ผู้ใช้งาน" : "Users",
      value: summary.distinctUsers ?? 0,
      hint: isThai ? "ผู้เรียนที่ไม่ซ้ำในตัวอย่าง" : "Distinct sampled learners",
      icon: Users,
    },
  ];

  return (
    <div className="border-t border-border">
      <div className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold">{isThai ? "ภาพรวมข้อมูลกิจกรรม" : "Activity stream overview"}</p>
            <p className="mt-1 max-w-2xl break-words text-xs text-muted-foreground">
              {isThai ? "แดชบอร์ดจากข้อมูล " : "Sample-based dashboard from "}
              <span className="font-medium">{sourceName}</span>.
              {summary.isFullAggregate
                ? (isThai ? " กราฟสรุปข้อมูลครบทั้งหมด ส่วนแถวในตารางเป็นตัวอย่าง" : "Charts summarize the full file scan; table rows are a preview.")
                : (isThai ? " ค่าต่าง ๆ สรุปจากระเบียนตัวอย่างที่เรียกมา" : "Values summarize retrieved sample records, not the full 11GB file.")}
            </p>
            <SourceDisclosure items={sourceItems} language={language} />
            <ReasoningDisclosure trace={activity?.decisionTrace} samples={sourceSampleRows} language={language} />
            </div>
          <div className="flex flex-wrap gap-2 text-xs">
            {sourceMeta?.object_type ? (
              <span className="rounded-full border bg-background px-3 py-1 font-medium uppercase">
                {sourceMeta.object_type}
              </span>
            ) : null}
            {sourceMeta?.size_bytes !== undefined ? (
              <span className="rounded-full border bg-background px-3 py-1 font-medium">
                {formatBytes(sourceMeta.size_bytes)}
              </span>
            ) : null}
            {sourceMeta?.sample_record_count ? (
              <span className="rounded-full border bg-background px-3 py-1 font-medium">
                {summary.isFullAggregate
                  ? `${formatNumber(summary.totalRecords)} ${isThai ? "แถวทั้งหมด" : "total rows"}`
                  : `${sourceMeta.sample_record_count} ${isThai ? "แถวตัวอย่าง" : "sampled rows"}`}
              </span>
            ) : null}
            {sourceMeta?.content_sample_ranges ? (
              <span className="rounded-full border bg-background px-3 py-1 font-medium">
                {sourceMeta.content_sample_ranges} {isThai ? "ช่วงข้อมูล" : "file ranges"}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      <div className="grid gap-3 border-t border-border p-4 sm:grid-cols-4">
        {stats.map((stat) => (
          <MetricCard
            key={stat.label}
            label={stat.label}
            value={stat.value}
            hint={stat.hint}
            icon={stat.icon}
          />
        ))}
      </div>

      <div className="grid gap-3 border-t border-border p-4 lg:grid-cols-2">
        {chartSlots
          .filter((slot) => slot.data?.length)
          .map((slot) => (
            <ChartSlotCard key={slot.id} slot={slot} />
          ))}
      </div>

      <div className="grid gap-3 border-t border-border p-4 md:grid-cols-3">
        <div className="rounded-lg border bg-background p-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Database className="h-4 w-4" aria-hidden="true" />
            {isThai ? "แหล่งข้อมูล" : "Source"}
          </div>
          <p className="mt-2 break-words text-sm font-medium">{sourceMeta?.s3_uri ?? sourceName}</p>
        </div>
        <div className="rounded-lg border bg-background p-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Clock className="h-4 w-4" aria-hidden="true" />
            {isThai ? "ช่วงเวลาของตัวอย่าง" : "Sample time window"}
          </div>
          <p className="mt-2 text-sm font-medium">{timeRange}</p>
        </div>
        <div className="rounded-lg border bg-background p-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Activity className="h-4 w-4" aria-hidden="true" />
            {isThai ? "กิจกรรมอันดับสูงสุด" : "Top event"}
          </div>
          <p className="mt-2 text-sm font-medium">
            {eventLeader
              ? `${humanLabel(eventLeader.label)} (${formatNumber(eventLeader.value)})`
              : (isThai ? "ไม่มีตัวอย่างกิจกรรม" : "No event sample")}
          </p>
        </div>
      </div>

      {records.length ? (
        <div className="border-t border-border px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {isThai ? "ระเบียนกิจกรรมตัวอย่าง" : "Sample activity records"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {isThai
              ? "ตัวอย่างระเบียนกิจกรรมที่อ่านได้ง่าย โดยย่อรหัสที่ยาวเมื่อจำเป็น"
              : "A readable slice of the retrieved activity records. Full identifiers are shortened where needed."}
          </p>
          <div className="mt-3 overflow-x-auto rounded-lg border">
            <table className="w-full min-w-[760px] text-left text-xs">
              <thead className="bg-muted/30 text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">{isThai ? "เวลา" : "Time"}</th>
                  <th className="px-3 py-2 font-medium">{isThai ? "แอป" : "App"}</th>
                  <th className="px-3 py-2 font-medium">{isThai ? "หมวดหมู่" : "Category"}</th>
                  <th className="px-3 py-2 font-medium">{isThai ? "กิจกรรม" : "Event"}</th>
                  <th className="px-3 py-2 font-medium">{isThai ? "หลักสูตร" : "Course"}</th>
                  <th className="px-3 py-2 font-medium">{isThai ? "ผู้ใช้งาน" : "User"}</th>
                </tr>
              </thead>
              <tbody>
                {records.slice(0, 8).map((record, index) => (
                  <tr key={`${record.source ?? "record"}-${record.index ?? index}`} className="border-t">
                    <td className="whitespace-nowrap px-3 py-2 align-top">
                      {formatDateTime(record["@timestamp"] ?? record.timestamp)}
                    </td>
                    <td className="px-3 py-2 align-top">{formatValue(record.appID)}</td>
                    <td className="px-3 py-2 align-top">{formatValue(record.eventCategory)}</td>
                    <td className="px-3 py-2 align-top">
                      <span className="rounded border bg-background px-2 py-1 font-medium">
                        {formatValue(record.event)}
                      </span>
                    </td>
                    <td className="max-w-[260px] break-words px-3 py-2 align-top" title={formatValue(record.courseID)}>
                      {courseName(record.courseID)}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2 align-top" title={formatValue(record.userID)}>
                      {shortId(record.userID, 6)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function PowerBiWidget({
  payload,
}: {
  payload: GraphDashboardWidgetPayload;
}) {
  const results = payload.results ?? [];
  const datasets = payload.datasets ?? [];
  const activity = payload.activity;
  const hasActivityDashboard = Boolean(
    (activity?.records?.length ?? 0) > 0 || Object.keys(activity?.charts ?? {}).length > 0,
  );
  const statusChart = payload.charts?.status ?? [
    { label: "Objects", value: payload.status?.objects ?? 0 },
    { label: "Nodes", value: payload.status?.nodes ?? 0 },
    { label: "Edges", value: payload.status?.edges ?? 0 },
  ];
  const resultChart =
    payload.charts?.results ??
    results.map((item) => ({
      label: humanLabel(item.label ?? item.path ?? item.source ?? "result"),
      score: item.score ?? 0,
    }));

  const stats = [
    { label: "Objects", value: payload.status?.objects ?? 0 },
    { label: "Nodes", value: payload.status?.nodes ?? 0 },
    { label: "Edges", value: payload.status?.edges ?? 0 },
  ];

  return (
    <section className="mt-3 overflow-hidden rounded-xl border border-border bg-background shadow-sm">
      {!hasActivityDashboard ? (
        <header className="border-b border-border bg-[#f3f2f1] px-4 py-3">
          <p className="text-sm font-semibold">{payload.title}</p>
          <p className="text-xs text-muted-foreground">
            Dashboard backed by retrieved graph context
          </p>
          <SourceDisclosure items={graphSourceItems(payload)} />
        </header>
      ) : null}

      <PromptDashboardSection activity={activity} />

      {!hasActivityDashboard ? (
        <>
          <div className="grid gap-3 border-t border-border p-4 sm:grid-cols-3">
            {stats.map((stat) => (
              <div key={stat.label} className="rounded-lg border bg-muted/20 p-3">
                <p className="text-xs text-muted-foreground">{stat.label}</p>
                <p className="mt-1 text-2xl font-semibold">{formatNumber(stat.value)}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-3 border-t border-border p-4 lg:grid-cols-2">
            <DashboardBarChart
              data={statusChart}
              dataKey="value"
              label="Graph size"
            />
            <DashboardBarChart
              data={resultChart}
              dataKey="score"
              label="Result relevance"
            />
          </div>
        </>
      ) : null}

      {!hasActivityDashboard ? (
        <div className="border-t border-border px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Data source
          </p>
          {datasets.length ? (
            <div className="mt-3 grid gap-3">
              {datasets.map((dataset) => (
                <article key={dataset.key} className="rounded-lg border bg-muted/10 p-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="break-words text-sm font-semibold">
                        {humanLabel(dataset.label ?? dataset.key)}
                      </p>
                      <p className="mt-1 break-words text-xs text-muted-foreground">
                        {dataset.s3Uri ?? dataset.key}
                      </p>
                    </div>
                    <div className="flex shrink-0 gap-2 text-xs">
                      {dataset.objectType ? (
                        <span className="rounded border bg-background px-2 py-1 font-medium uppercase">
                          {dataset.objectType}
                        </span>
                      ) : null}
                      {dataset.sizeBytes !== undefined ? (
                        <span className="rounded border bg-background px-2 py-1 font-medium">
                          {formatBytes(dataset.sizeBytes)}
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
                    {dataset.bucket ? (
                      <div>
                        <dt className="text-muted-foreground">Bucket</dt>
                        <dd className="break-words font-medium">{dataset.bucket}</dd>
                      </div>
                    ) : null}
                    {dataset.lastModified ? (
                      <div>
                        <dt className="text-muted-foreground">Last modified</dt>
                        <dd className="break-words font-medium">{dataset.lastModified}</dd>
                      </div>
                    ) : null}
                    {dataset.ETAG ? (
                      <div>
                        <dt className="text-muted-foreground">ETAG</dt>
                        <dd className="break-words font-medium">{dataset.ETAG}</dd>
                      </div>
                    ) : null}
                    {dataset.matchedFields?.length ? (
                      <div>
                        <dt className="text-muted-foreground">Fields used in this view</dt>
                        <dd className="break-words font-medium">
                          {dataset.matchedFields.slice(0, 6).map((field) => humanLabel(field)).join(", ")}
                        </dd>
                      </div>
                    ) : null}
                  </dl>
                </article>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-sm text-muted-foreground">
              No dataset metadata was found for this request.
            </p>
          )}
        </div>
      ) : null}

      {!hasActivityDashboard ? (
        <div className="border-t border-border px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Matching graph nodes
          </p>
          {results.length ? (
            <div className="mt-2 space-y-2">
              {results.map((item, index) => (
                <div key={`${item.id ?? item.path ?? index}`} className="text-sm">
                  <p className="font-medium">
                    {humanLabel(item.label ?? item.path ?? item.source ?? "Graph node")}
                  </p>
                  <p className="break-words text-xs text-muted-foreground">
                    {item.source ?? item.path}
                  </p>
                  <p className="mt-1 break-words text-xs">
                    {previewText(item.value, item.text)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-sm text-muted-foreground">
              No matching graph nodes were found for this request.
            </p>
          )}
        </div>
      ) : null}
    </section>
  );
}
