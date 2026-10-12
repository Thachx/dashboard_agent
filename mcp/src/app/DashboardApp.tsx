import { useEffect, useMemo, useState } from "react";
import { App } from "@modelcontextprotocol/ext-apps";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  Treemap,
  XAxis,
  YAxis,
} from "recharts";

import {
  DashboardDocumentSchema,
  type DashboardChart,
  type DashboardDocument,
} from "../shared/dashboard-contract";
import { demoDashboard } from "../shared/demo-dashboard";

const COLORS = ["#2f61df", "#18a873", "#f07a32", "#805ad5", "#ca3f64", "#2196a6"];

function formatNumber(value: number | string): string {
  if (typeof value === "string") return value;
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function chartRows(chart: DashboardChart): Array<Record<string, string | number>> {
  if (!chart.data.some((item) => item.series)) {
    return chart.data.map((item) => ({
      label: item.label,
      value: item.value ?? 0,
      score: item.score ?? 0,
    }));
  }
  const rows = new Map<string, Record<string, string | number>>();
  for (const item of chart.data) {
    const row = rows.get(item.label) ?? { label: item.label };
    row[item.series ?? "Value"] = item.value ?? item.score ?? 0;
    rows.set(item.label, row);
  }
  return [...rows.values()];
}

function ChartView({ chart }: { chart: DashboardChart }) {
  const rows = chartRows(chart);
  const series = [...new Set(chart.data.map((item) => item.series).filter(Boolean))] as string[];
  const type = chart.chartType.toLowerCase().replaceAll("-", "_");
  const dataKey = series.length ? undefined : chart.data.some((item) => item.value != null) ? "value" : "score";
  const tooltip = <Tooltip formatter={(value) => formatNumber(Number(value))} />;

  if (type.includes("donut") || type === "pie") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={chart.data} dataKey={dataKey ?? "value"} nameKey="label" innerRadius="48%" outerRadius="74%" paddingAngle={1} isAnimationActive={false}>
            {chart.data.map((item, index) => <Cell key={`${item.label}-${index}`} fill={COLORS[index % COLORS.length]} />)}
          </Pie>
          {tooltip}
          <Legend verticalAlign="bottom" align="center" layout="horizontal" />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (type.includes("treemap")) {
    const tree = chart.data.map((item) => ({ name: item.label, size: item.value ?? item.score ?? 0 }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <Treemap data={tree} dataKey="size" nameKey="name" stroke="#fff" fill={COLORS[0]} aspectRatio={16 / 7} isAnimationActive={false}>
          {tooltip}
        </Treemap>
      </ResponsiveContainer>
    );
  }

  if (type.includes("line") || type.includes("area") || type.includes("time")) {
    const Chart = type.includes("area") ? AreaChart : LineChart;
    return (
      <ResponsiveContainer width="100%" height="100%">
        <Chart data={rows} margin={{ top: 8, right: 18, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" minTickGap={28} tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={formatNumber} width={48} tick={{ fontSize: 11 }} />
          {tooltip}
          {series.length > 1 && <Legend />}
          {(series.length ? series : [dataKey ?? "value"]).map((key, index) => type.includes("area")
            ? <Area key={key} dataKey={key} type="monotone" stroke={COLORS[index % COLORS.length]} fill={COLORS[index % COLORS.length]} fillOpacity={0.16} isAnimationActive={false} />
            : <Line key={key} dataKey={key} type="monotone" stroke={COLORS[index % COLORS.length]} strokeWidth={2.5} dot={false} isAnimationActive={false} />)}
        </Chart>
      </ResponsiveContainer>
    );
  }

  if (type.includes("horizontal") || type.includes("rank")) {
    const height = Math.max(220, rows.length * 38);
    return (
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 20, left: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={formatNumber} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
            {tooltip}
            {series.length > 1 && <Legend />}
            {(series.length ? series : [dataKey ?? "value"]).map((key, index) => (
              <Bar key={key} dataKey={key} stackId={series.length > 1 ? "total" : undefined} fill={COLORS[index % COLORS.length]} radius={series.length > 1 ? 0 : [0, 4, 4, 0]} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rows} margin={{ top: 8, right: 18, left: 0, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" interval={0} height={46} tick={{ fontSize: 10 }} />
        <YAxis tickFormatter={formatNumber} width={48} tick={{ fontSize: 11 }} />
        {tooltip}
        {series.length > 1 && <Legend />}
        {(series.length ? series : [dataKey ?? "value"]).map((key, index) => (
          <Bar key={key} dataKey={key} stackId={type.includes("stack") ? "total" : undefined} fill={COLORS[index % COLORS.length]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function Dashboard({ dashboard }: { dashboard: DashboardDocument }) {
  const hasVisuals = dashboard.metrics.length > 0 || dashboard.charts.length > 0;
  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Dashboard Agent</p>
          <h1>{dashboard.title}</h1>
          {dashboard.subtitle && <p className="subtitle">{dashboard.subtitle}</p>}
        </div>
        <span className="execution-badge">{dashboard.execution.source}</span>
      </header>

      {!hasVisuals && (
        <section className="answer-panel" aria-label="Analysis answer">
          <div>{dashboard.summary}</div>
        </section>
      )}

      {dashboard.metrics.length > 0 && (
        <section className="metric-grid" aria-label="Summary metrics">
          {dashboard.metrics.map((metric) => (
            <article className="metric" key={metric.id}>
              <span>{metric.label}</span>
              <strong>{formatNumber(metric.value)}</strong>
              {metric.hint && <small>{metric.hint}</small>}
            </article>
          ))}
        </section>
      )}

      {dashboard.charts.length > 0 && (
        <section className="chart-grid" aria-label="Dashboard charts">
          {dashboard.charts.map((chart) => (
            <article className={`chart chart-${chart.chartType.toLowerCase()}`} key={chart.id}>
              <div className="chart-heading">
                <h2>{chart.title}</h2>
                {chart.reason && <p>{chart.reason}</p>}
              </div>
              <div className="chart-canvas"><ChartView chart={chart} /></div>
            </article>
          ))}
        </section>
      )}

      {(dashboard.sources.length > 0 || dashboard.reasoningSummary.length > 0) && (
        <footer className="dashboard-details">
          {dashboard.sources.length > 0 && (
            <details>
              <summary>Sources ({dashboard.sources.length})</summary>
              <ul>{dashboard.sources.map((source) => <li key={source.id}>{source.label}{source.path ? ` - ${source.path}` : ""}</li>)}</ul>
            </details>
          )}
          {dashboard.reasoningSummary.length > 0 && (
            <details>
              <summary>Analysis summary</summary>
              <dl>{dashboard.reasoningSummary.map((item, index) => <div key={`${item.stage}-${index}`}><dt>{item.stage}</dt><dd>{item.detail}</dd></div>)}</dl>
            </details>
          )}
        </footer>
      )}
    </main>
  );
}

export function DashboardApp() {
  const [dashboard, setDashboard] = useState<DashboardDocument>(demoDashboard);
  const isEmbedded = window.parent !== window;

  useEffect(() => {
    if (!isEmbedded) return;
    const app = new App({ name: "Dashboard Agent view", version: "0.1.0" });
    app.ontoolresult = (result) => {
      const parsed = DashboardDocumentSchema.safeParse(result.structuredContent);
      if (parsed.success) setDashboard(parsed.data);
    };
    void app.connect();
    return () => { void app.close(); };
  }, [isEmbedded]);

  const validated = useMemo(() => DashboardDocumentSchema.parse(dashboard), [dashboard]);
  return <Dashboard dashboard={validated} />;
}
