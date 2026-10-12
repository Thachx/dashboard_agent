import { z } from "zod";

export const ChartDatumSchema = z.object({
  label: z.string(),
  value: z.number().optional(),
  score: z.number().optional(),
  series: z.string().optional(),
  rawLabel: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export const DashboardMetricSchema = z.object({
  id: z.string(),
  label: z.string(),
  value: z.union([z.number(), z.string()]),
  hint: z.string().optional(),
});

export const DashboardChartSchema = z.object({
  id: z.string(),
  title: z.string(),
  chartType: z.string(),
  field: z.string().optional().default("value"),
  reason: z.string().optional(),
  data: z.array(ChartDatumSchema),
});

export const DashboardSourceSchema = z.object({
  id: z.string(),
  label: z.string(),
  path: z.string().optional(),
  objectType: z.string().optional(),
  matchedFields: z.array(z.string()).optional(),
});

export const DashboardLayoutBlockSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("metric"), id: z.string(), span: z.number().optional() }),
  z.object({ type: z.literal("chart"), slotId: z.string(), span: z.number().optional() }),
  z.object({ type: z.literal("records"), span: z.number().optional() }),
]);

export const DashboardDocumentSchema = z.object({
  schemaVersion: z.literal("1.0"),
  dashboardId: z.string(),
  query: z.string(),
  title: z.string(),
  subtitle: z.string().optional(),
  locale: z.string().default("en"),
  timezone: z.string().default("UTC"),
  generatedAt: z.string(),
  summary: z.string(),
  metrics: z.array(DashboardMetricSchema),
  charts: z.array(DashboardChartSchema),
  layout: z.array(DashboardLayoutBlockSchema),
  sources: z.array(DashboardSourceSchema),
  reasoningSummary: z.array(z.object({ stage: z.string(), detail: z.string() })),
  execution: z.object({
    source: z.enum(["graph", "duckdb", "hybrid", "fixture", "unknown"]),
    graphUsed: z.boolean(),
    duckdbUsed: z.boolean(),
  }),
});

export type ChartDatum = z.infer<typeof ChartDatumSchema>;
export type DashboardChart = z.infer<typeof DashboardChartSchema>;
export type DashboardDocument = z.infer<typeof DashboardDocumentSchema>;
export type DashboardLayoutBlock = z.infer<typeof DashboardLayoutBlockSchema>;

type UnknownRecord = Record<string, unknown>;

const METRIC_KEYS = [
  "totalRecords",
  "sampleRecords",
  "distinctEvents",
  "distinctCourses",
  "distinctUsers",
  "distinctDimensionValues",
  "totalDistinctMeasure",
  "topDimensionValue",
] as const;

const DEFAULT_METRIC_LABELS: Record<string, string> = {
  totalRecords: "Activity rows",
  sampleRecords: "Sample rows",
  distinctEvents: "Event types",
  distinctCourses: "Courses",
  distinctUsers: "Users",
  distinctDimensionValues: "Groups",
  totalDistinctMeasure: "Matched values",
  topDimensionValue: "Leading value",
};

function record(value: unknown): UnknownRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : {};
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function numeric(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  return undefined;
}

function normalizeChartData(value: unknown): Array<z.infer<typeof ChartDatumSchema>> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item, index) => {
    const source = record(item);
    const label = text(source.label ?? source.name ?? source.key, `Item ${index + 1}`);
    const datum = {
      label,
      value: numeric(source.value),
      score: numeric(source.score),
      series: text(source.series) || undefined,
      rawLabel: text(source.rawLabel) || undefined,
    };
    return datum.value == null && datum.score == null ? [] : [datum];
  });
}

function normalizeSources(payload: UnknownRecord, activity: UnknownRecord) {
  const sourceValues: unknown[] = [];
  if (Array.isArray(payload.datasets)) sourceValues.push(...payload.datasets);
  const activityDatasets = activity.datasets;
  if (Array.isArray(activityDatasets)) sourceValues.push(...activityDatasets);
  else if (activityDatasets && typeof activityDatasets === "object") {
    sourceValues.push(...Object.values(activityDatasets));
  }

  const unique = new Map<string, z.infer<typeof DashboardSourceSchema>>();
  for (const value of sourceValues) {
    const source = record(value);
    const path = text(source.path ?? source.source ?? source.key ?? source.s3Uri);
    const id = text(source.id ?? source.key ?? path);
    if (!id) continue;
    const matchedFields = Array.isArray(source.matchedFields)
      ? source.matchedFields.filter((field): field is string => typeof field === "string")
      : undefined;
    unique.set(id, {
      id,
      label: text(source.label ?? source.sourceLabel ?? source.fileLabel, id),
      path: path || undefined,
      objectType: text(source.objectType ?? source.object_type) || undefined,
      matchedFields,
    });
  }
  return [...unique.values()];
}

function inferExecution(activity: UnknownRecord): DashboardDocument["execution"] {
  const summary = record(activity.summary);
  const source = `${text(summary.executionSource)} ${text(summary.source)} ${text(activity.executionSource)}`.toLowerCase();
  const graphUsed = source.includes("graph") || Boolean(activity.graphContext);
  const duckdbUsed = source.includes("duckdb") || source.includes("dashboard_agent_");
  if (graphUsed && duckdbUsed) return { source: "hybrid", graphUsed, duckdbUsed };
  if (graphUsed) return { source: "graph", graphUsed, duckdbUsed };
  if (duckdbUsed) return { source: "duckdb", graphUsed, duckdbUsed };
  return { source: "unknown", graphUsed: false, duckdbUsed: false };
}

export function dashboardFromGraphPayload(
  payloadValue: unknown,
  options: { query: string; locale?: string; timezone?: string; dashboardId?: string },
): DashboardDocument {
  const payload = record(payloadValue);
  const activity = record(payload.activity);
  const summary = record(activity.summary);
  const layoutSpec = record(activity.layoutSpec);
  const metricLabels = record(summary.metricLabels);

  const metrics = METRIC_KEYS.flatMap((id) => {
    const value = summary[id];
    if (value == null || (typeof value !== "number" && typeof value !== "string")) return [];
    return [{
      id,
      label: text(metricLabels[id], DEFAULT_METRIC_LABELS[id]),
      value,
      hint: id.startsWith("distinct") || id.startsWith("totalDistinct")
        ? "Distinct values included in the aggregate"
        : undefined,
    }];
  });

  const charts = (Array.isArray(activity.chartSlots) ? activity.chartSlots : [])
    .flatMap((value, index) => {
      const slot = record(value);
      const data = normalizeChartData(slot.data);
      if (!data.length) return [];
      return [{
        id: text(slot.id, `chart-${index + 1}`),
        title: text(slot.title, `Chart ${index + 1}`),
        chartType: text(slot.chartType, "bar"),
        field: text(slot.field, "value"),
        reason: text(slot.reason) || undefined,
        data,
      }];
    });

  const chartIds = new Set(charts.map((chart) => chart.id));
  const metricIds = new Set<string>(metrics.map((metric) => metric.id));
  const layout: DashboardLayoutBlock[] = [];
  for (const value of Array.isArray(layoutSpec.blocks) ? layoutSpec.blocks : []) {
    const block = record(value);
    const type = text(block.type);
    if (type === "metric" && metricIds.has(text(block.id))) {
      layout.push({ type: "metric", id: text(block.id), span: numeric(block.span) });
    } else if (type === "chart" && chartIds.has(text(block.slotId))) {
      layout.push({ type: "chart", slotId: text(block.slotId), span: numeric(block.span) });
    } else if (type === "records") {
      layout.push({ type: "records", span: numeric(block.span) });
    }
  }

  const finalLayout = layout.length
    ? layout
    : [
        ...metrics.map((metric) => ({ type: "metric" as const, id: metric.id })),
        ...charts.map((chart) => ({ type: "chart" as const, slotId: chart.id })),
      ];

  const reasoningSummary = (Array.isArray(activity.decisionTrace) ? activity.decisionTrace : [])
    .flatMap((value) => {
      const item = record(value);
      const detail = text(item.detail ?? item.summary ?? item.message);
      if (!detail) return [];
      return [{ stage: text(item.stage ?? item.name, "Analysis"), detail }];
    });

  const title = text(layoutSpec.title ?? payload.title, "Dashboard");
  const subtitle = text(layoutSpec.subtitle) || undefined;
  const topChart = charts[0];
  const summaryText = subtitle
    ?? (topChart ? `${topChart.title} with ${topChart.data.length} plotted values.` : `${metrics.length} dashboard metrics.`);

  return DashboardDocumentSchema.parse({
    schemaVersion: "1.0",
    dashboardId: options.dashboardId ?? crypto.randomUUID(),
    query: options.query,
    title,
    subtitle,
    locale: options.locale ?? "en",
    timezone: options.timezone ?? "UTC",
    generatedAt: new Date().toISOString(),
    summary: summaryText,
    metrics,
    charts,
    layout: finalLayout,
    sources: normalizeSources(payload, activity),
    reasoningSummary,
    execution: inferExecution(activity),
  });
}

export function dashboardFromTextResponse(
  responseText: string,
  options: { query: string; locale?: string; timezone?: string; dashboardId?: string },
): DashboardDocument {
  const normalized = responseText.replaceAll("\r\n", "\n").trim();
  if (!normalized) throw new Error("Dashboard Agent returned an empty response.");

  const lines = normalized.split("\n");
  const heading = lines[0]?.match(/^#{1,6}\s+(.+?)\s*$/);
  const title = heading?.[1]?.trim() || "Analysis result";
  const summary = (heading ? lines.slice(1).join("\n").trim() : normalized) || title;

  return DashboardDocumentSchema.parse({
    schemaVersion: "1.0",
    dashboardId: options.dashboardId ?? crypto.randomUUID(),
    query: options.query,
    title,
    locale: options.locale ?? "en",
    timezone: options.timezone ?? "UTC",
    generatedAt: new Date().toISOString(),
    summary,
    metrics: [],
    charts: [],
    layout: [],
    sources: [],
    reasoningSummary: [{
      stage: "Answer",
      detail: "Returned the upstream analytical response without creating an unsupported chart.",
    }],
    execution: { source: "unknown", graphUsed: false, duckdbUsed: false },
  });
}
