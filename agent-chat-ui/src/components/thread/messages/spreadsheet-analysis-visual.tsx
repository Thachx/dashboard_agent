import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  Rectangle,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type SpreadsheetChartPoint = {
  label: string;
  value: number;
  percent?: number;
  fill?: string;
};

type SpreadsheetChart = {
  id: string;
  title: string;
  chartType: "bar" | "pie";
  total: number;
  series: SpreadsheetChartPoint[];
  summary?: string;
  reason?: string;
};

type SpreadsheetInsight = {
  title: string;
  summary: string;
};

export type SpreadsheetAnalysisVisualPayload = {
  version: number;
  kind: "spreadsheet-analysis-visual";
  displayMode?: "default" | "single-chart";
  userLanguage?: "th" | "en";
  spreadsheetId: string;
  spreadsheetTitle: string;
  spreadsheetUrl: string;
  processedSheetName: string;
  detailSheetName: string;
  summarySheetName: string;
  rowCountWritten: number;
  questionCount: number;
  analysisRequest?: string;
  insights?: SpreadsheetInsight[];
  charts: SpreadsheetChart[];
};

const CHART_COLORS = [
  "#2563eb",
  "#f59e0b",
  "#10b981",
  "#ef4444",
  "#8b5cf6",
  "#06b6d4",
  "#f97316",
  "#84cc16",
];

function formatPercent(value: number | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return "";
  return `${value.toFixed(1)}%`;
}

function formatTooltipValue(
  value: number,
  name: string,
  props: { payload?: SpreadsheetChartPoint },
) {
  const percent = formatPercent(props.payload?.percent);
  return [`${value}${percent ? ` (${percent})` : ""}`, name];
}

function isThaiPayload(payload: SpreadsheetAnalysisVisualPayload): boolean {
  return payload.userLanguage === "th";
}

function isSingleExplicitChartPayload(
  payload: SpreadsheetAnalysisVisualPayload,
): boolean {
  if (payload.displayMode === "single-chart") return true;
  if ((payload.insights?.length ?? 0) > 0) return false;
  if (payload.charts.length !== 1) return false;

  const brief = String(payload.analysisRequest ?? "").toLowerCase();
  return (
    brief.includes("distribution graph") ||
    brief.includes("distribution chart") ||
    brief.includes("score distribution") ||
    brief.includes("histogram") ||
    brief.includes("การกระจายคะแนน") ||
    brief.includes("กราฟการกระจาย") ||
    brief.includes("กราฟกระจาย")
  );
}

export function SpreadsheetAnalysisVisual({
  payload,
}: {
  payload: SpreadsheetAnalysisVisualPayload;
}) {
  if (!payload.charts.length && !payload.insights?.length) return null;
  const preferThai = isThaiPayload(payload);
  const singleChartMode =
    payload.charts.length === 1 || isSingleExplicitChartPayload(payload);

  return (
    <section
      className={
        singleChartMode
          ? "mt-3 w-full min-w-0"
          : "mt-3 w-full min-w-0 rounded-lg border border-border bg-muted/20"
      }
    >
      {!singleChartMode ? <div className="border-b border-border px-4 py-3">
        <div className="text-sm font-medium">
          {preferThai ? "กราฟวิเคราะห์สเปรดชีต" : "Spreadsheet graphs"}
        </div>
        <div className="mt-1 text-xs text-muted-foreground">
          {preferThai
            ? `${payload.questionCount} คำถามที่มีคำตอบ, ${payload.rowCountWritten} แถวที่ใช้วิเคราะห์`
            : `${payload.questionCount} questions with responses, ${payload.rowCountWritten} analyzed rows`}
        </div>
      </div> : null}
      <div className={singleChartMode ? "space-y-4" : "space-y-4 p-4"}>
        {payload.insights?.length ? (
          <section className="min-w-0 rounded-md border border-border bg-background px-3 py-3">
            <div className="mb-3 text-sm font-medium">
              {preferThai ? "การวิเคราะห์เชิงลึก" : "Deep analysis"}
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {payload.insights.map((insight, index) => (
                <div
                  key={`${insight.title}-${index}`}
                  className="rounded-md border border-border/70 bg-muted/20 px-3 py-3"
                >
                  <div className="text-sm font-medium">{insight.title}</div>
                  <div className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {insight.summary}
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : null}
        {payload.charts.map((chart) => {
          const chartData = chart.series.map((point, index) => ({
            ...point,
            fill: CHART_COLORS[index % CHART_COLORS.length],
          }));

          return (
            <section
              key={chart.id}
              className="min-w-0 rounded-md border border-border bg-background px-4 py-4"
            >
              <div className="space-y-5">
                <div className="min-w-0">
                  <div className="mb-2 text-sm font-medium leading-snug">
                    {chart.title}
                  </div>
                  <div className="mb-3 text-xs text-muted-foreground">
                    {preferThai ? `${chart.total} คำตอบ` : `${chart.total} answers`}
                  </div>
                  {chart.summary ? (
                    <div className="mb-4 text-sm leading-relaxed text-foreground/85">
                      {chart.summary}
                    </div>
                  ) : null}
                  <div className="space-y-2">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {preferThai ? "สรุป" : "Analysis"}
                    </div>
                    <div className="space-y-1">
                      {chartData.slice(0, 6).map((point, index) => (
                        <div
                          key={`${chart.id}-legend-${point.label}`}
                          className="flex items-start gap-2 text-xs"
                        >
                          <span
                            className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{
                              backgroundColor:
                                point.fill ??
                                CHART_COLORS[index % CHART_COLORS.length],
                            }}
                          />
                          <span className="min-w-0 flex-1 break-words text-foreground/90">
                            {point.label}
                          </span>
                          <span className="shrink-0 text-muted-foreground">
                            {point.value}
                            {point.percent !== undefined
                              ? ` · ${formatPercent(point.percent)}`
                              : ""}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="h-96 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      {chart.chartType === "pie" ? (
                        <PieChart>
                          <Tooltip formatter={formatTooltipValue} />
                          <Pie
                            data={chartData}
                            dataKey="value"
                            nameKey="label"
                            innerRadius={70}
                            outerRadius={130}
                            paddingAngle={2}
                          >
                            {chartData.map((entry, index) => (
                              <Cell
                                key={`${chart.id}-${entry.label}`}
                                fill={
                                  entry.fill ??
                                  CHART_COLORS[index % CHART_COLORS.length]
                                }
                              />
                            ))}
                          </Pie>
                        </PieChart>
                      ) : (
                        <BarChart
                          data={chartData}
                          margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
                        >
                          <CartesianGrid vertical={false} strokeDasharray="3 3" />
                          <XAxis
                            type="category"
                            dataKey="label"
                            interval={0}
                            tick={{ fontSize: 12 }}
                            angle={chartData.length > 8 ? -25 : 0}
                            textAnchor={chartData.length > 8 ? "end" : "middle"}
                            height={chartData.length > 8 ? 84 : 48}
                          />
                          <YAxis
                            type="number"
                            allowDecimals={false}
                          />
                          <Tooltip formatter={formatTooltipValue} />
                          <Bar
                            dataKey="value"
                            radius={[4, 4, 0, 0]}
                            shape={(props: any) => (
                              <Rectangle
                                {...props}
                                fill={props.payload?.fill ?? CHART_COLORS[0]}
                              />
                            )}
                          />
                        </BarChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
