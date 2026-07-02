"use client";

import {
  Activity,
  BarChart3,
  BookOpen,
  Clock,
  Database,
  ListChecks,
  MousePointerClick,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ChartDatum = {
  label: string;
  value?: number;
  score?: number;
};

const CHART_COLORS = ["#2563eb", "#10b981", "#f97316", "#7c3aed", "#ec4899", "#64748b"];

type DatasetSummary = {
  key: string;
  label?: string;
  s3Uri?: string;
  bucket?: string;
  objectType?: string;
  sizeBytes?: number | string;
  etag?: string;
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
  return value.replace(/[_-]/g, " ");
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
      <p className="mt-2 text-xs text-muted-foreground">{hint}</p>
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
      <div className="mt-3 grid gap-3 sm:grid-cols-[140px_1fr]">
        <div className="relative h-36">
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
  if (slot.chartType === "line") {
    return <TimelineChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "stat") {
    return <StatSlotCard data={slot.data} title={slot.title} field={slot.field} />;
  }
  if (slot.chartType === "donut") {
    return <DonutChartCard data={slot.data} title={slot.title} />;
  }
  if (slot.chartType === "column") {
    return <MiniColumnChart data={slot.data} title={slot.title} />;
  }
  return <BreakdownList data={slot.data} title={slot.title} kind={kind} />;
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
  const metricDefs: Record<
    string,
    { label: string; value: number | string | undefined; hint: string; icon: typeof Activity }
  > = {
    totalRecords: {
      label: "Activity rows",
      value: summary.totalRecords ?? sourceMeta?.totalRecords ?? summary.sampleRecords ?? records.length,
      hint: summary.isFullAggregate ? "Full aggregate" : "Matched sample",
      icon: Activity,
    },
    sampleRecords: {
      label: "Sample rows",
      value: summary.sampleRecords ?? records.length,
      hint: "Rows available in the widget",
      icon: Database,
    },
    distinctEvents: {
      label: "Event types",
      value: summary.distinctEvents ?? charts.events?.length,
      hint: "Distinct event values",
      icon: BarChart3,
    },
    distinctCourses: {
      label: "Courses",
      value: summary.distinctCourses ?? charts.courses?.length,
      hint: "Distinct course values",
      icon: BookOpen,
    },
    distinctUsers: {
      label: "Users",
      value: summary.distinctUsers,
      hint: "Distinct user values",
      icon: Users,
    },
    distinctDimensionValues: {
      label: `${formatValue((summary as Record<string, unknown>).topDimensionName, "Dimension")} values`,
      value: (summary as Record<string, unknown>).distinctDimensionValues as number | string | undefined,
      hint: "Distinct ranked dimension values",
      icon: Database,
    },
    totalDistinctMeasure: {
      label: `Total ${formatValue((summary as Record<string, unknown>).measureName, "measure")}`,
      value: (summary as Record<string, unknown>).totalDistinctMeasure as number | string | undefined,
      hint: "Distinct values included in the aggregate",
      icon: Users,
    },
    topDimensionValue: {
      label: `Top ${formatValue((summary as Record<string, unknown>).measureName, "measure")}`,
      value: (summary as Record<string, unknown>).topDimensionValue as number | string | undefined,
      hint: formatValue((summary as Record<string, unknown>).topDimensionLabel, "Top value"),
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
              <th className="px-3 py-2 font-medium">Time</th>
              <th className="px-3 py-2 font-medium">Event</th>
              <th className="px-3 py-2 font-medium">Course</th>
              <th className="px-3 py-2 font-medium">User</th>
              <th className="px-3 py-2 font-medium">Context</th>
            </tr>
          </thead>
          <tbody>
            {records.slice(0, 8).map((record, index) => (
              <tr key={`${record.timestamp ?? record["@timestamp"] ?? index}`} className="border-t border-border">
                <td className="px-3 py-2">{formatDateTime(record["@timestamp"] ?? record.timestamp)}</td>
                <td className="px-3 py-2">{String(record.event ?? record.event_type ?? "Activity")}</td>
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

  return (
    <section className="overflow-hidden bg-background">
      <div className="border-b border-border p-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-base font-semibold">{layoutSpec.title ?? "Activity dashboard"}</h3>
          {layoutSpec.subtitle ? (
            <p className="max-w-3xl text-sm text-muted-foreground">{layoutSpec.subtitle}</p>
          ) : null}
        </div>
      </div>

      <div className="grid gap-3 p-4 lg:grid-cols-[minmax(180px,0.8fr)_minmax(360px,2.2fr)]">
        <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-1">
          {metricBlocks.map((block, index) => {
          if (block.type === "metric") {
            const metric = metricDefs[block.id];
            if (!metric || metric.value == null) return null;
            return (
              <div key={`${block.type}-${block.id}-${index}`} className="min-w-0">
                <MetricCard
                  label={metric.label}
                  value={metric.value}
                  hint={metric.hint}
                  icon={metric.icon}
                />
              </div>
            );
          }
          return null;
        })}
        </div>
        <div className="min-w-0">
          {chartBlocks.map((block, index) => {
          if (block.type === "chart") {
            const slot = slotById.get(block.slotId);
            if (!slot?.data?.length) return null;
            return (
              <div key={`${block.type}-${block.slotId}-${index}`} className="min-w-0">
                <ChartSlotCard slot={slot} />
              </div>
            );
          }
          return null;
        })}
        </div>
        {recordBlocks.map((block, index) => {
          if (block.type === "records") {
            return <div key={`${block.type}-${index}`} className="min-w-0 lg:col-span-2">{renderRecords()}</div>;
          }
          return null;
        })}
      </div>
    </section>
  );
}

function ActivityDashboardSection({ activity }: { activity?: ActivityDashboardPayload }) {
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
  const sourceMeta = Object.values(activity?.datasets ?? {})[0];
  const sourceName = sourceMeta?.key ?? records[0]?.source ?? "Matched activity data";
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
        : "No timestamp sample";
  if (!records.length && !Object.keys(charts).length) return null;

  const stats = [
    {
      label: summary.isFullAggregate ? "Total activity rows" : "Sampled rows",
      value: summary.totalRecords ?? summary.sampleRecords ?? records.length,
      hint: summary.isFullAggregate ? "Full file scan" : "Rows read from the activity stream",
      icon: ListChecks,
    },
    {
      label: "Event types",
      value: summary.distinctEvents ?? charts.events?.length ?? 0,
      hint: "Unique actions in the sample",
      icon: MousePointerClick,
    },
    {
      label: "Courses",
      value: summary.distinctCourses ?? charts.courses?.length ?? 0,
      hint: "Course IDs represented",
      icon: BookOpen,
    },
    {
      label: "Users",
      value: summary.distinctUsers ?? 0,
      hint: "Distinct sampled learners",
      icon: Users,
    },
  ];

  return (
    <div className="border-t border-border">
      <div className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold">Activity stream overview</p>
            <p className="mt-1 max-w-2xl break-words text-xs text-muted-foreground">
              Sample-based dashboard from <span className="font-medium">{sourceName}</span>.
              {summary.isFullAggregate
                ? "Charts summarize the full file scan; table rows are a preview."
                : "Values summarize retrieved sample records, not the full 11GB file."}
            </p>
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
                {summary.isFullAggregate ? `${formatNumber(summary.totalRecords)} total rows` : `${sourceMeta.sample_record_count} sampled rows`}
              </span>
            ) : null}
            {sourceMeta?.content_sample_ranges ? (
              <span className="rounded-full border bg-background px-3 py-1 font-medium">
                {sourceMeta.content_sample_ranges} file ranges
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
            Source
          </div>
          <p className="mt-2 break-words text-sm font-medium">{sourceMeta?.s3_uri ?? sourceName}</p>
        </div>
        <div className="rounded-lg border bg-background p-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Clock className="h-4 w-4" aria-hidden="true" />
            Sample time window
          </div>
          <p className="mt-2 text-sm font-medium">{timeRange}</p>
        </div>
        <div className="rounded-lg border bg-background p-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Activity className="h-4 w-4" aria-hidden="true" />
            Top event
          </div>
          <p className="mt-2 text-sm font-medium">
            {eventLeader ? `${eventLeader.label} (${formatNumber(eventLeader.value)})` : "No event sample"}
          </p>
        </div>
      </div>

      {records.length ? (
        <div className="border-t border-border px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Sample activity records
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            A readable slice of the retrieved activity records. Full identifiers are shortened where needed.
          </p>
          <div className="mt-3 overflow-x-auto rounded-lg border">
            <table className="w-full min-w-[760px] text-left text-xs">
              <thead className="bg-muted/30 text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">Time</th>
                  <th className="px-3 py-2 font-medium">App</th>
                  <th className="px-3 py-2 font-medium">Category</th>
                  <th className="px-3 py-2 font-medium">Event</th>
                  <th className="px-3 py-2 font-medium">Course</th>
                  <th className="px-3 py-2 font-medium">User</th>
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
      label: item.label ?? item.path ?? item.source ?? "result",
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
                        {dataset.label ? dataset.label.replace(/[-_]/g, " ") : dataset.key}
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
                    {dataset.etag ? (
                      <div>
                        <dt className="text-muted-foreground">ETag</dt>
                        <dd className="break-words font-medium">{dataset.etag}</dd>
                      </div>
                    ) : null}
                    {dataset.matchedFields?.length ? (
                      <div>
                        <dt className="text-muted-foreground">Fields used in this view</dt>
                        <dd className="break-words font-medium">
                          {dataset.matchedFields.slice(0, 6).join(", ")}
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
                    {item.label ?? item.path ?? item.source ?? "Graph node"}
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
