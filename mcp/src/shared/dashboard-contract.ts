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
