import { existsSync } from "node:fs";
import { createRequire } from "node:module";

import { createCanvas, GlobalFonts } from "@napi-rs/canvas";
import { Resvg } from "@resvg/resvg-js";

import type { DashboardChart, DashboardDocument } from "../shared/dashboard-contract.js";

const WIDTH = 1200;
const COLORS = ["#2f61df", "#18a873", "#f07a32", "#805ad5", "#ca3f64"];
const require = createRequire(import.meta.url);
const sarabunRegularFontFile = require.resolve("@expo-google-fonts/sarabun/400Regular/Sarabun_400Regular.ttf");
const sarabunSemiBoldFontFile = require.resolve("@expo-google-fonts/sarabun/600SemiBold/Sarabun_600SemiBold.ttf");
const sarabunBoldFontFile = require.resolve("@expo-google-fonts/sarabun/700Bold/Sarabun_700Bold.ttf");
const configuredLabelFontFile = process.env.DASHBOARD_PREVIEW_LABEL_FONT_FILE?.trim();
const labelFontFamily = (process.env.DASHBOARD_PREVIEW_LABEL_FONT_FAMILY ?? "Sarabun")
  .replaceAll(/["\\\r\n]/g, "")
  .trim() || "Sarabun";
const configuredLabelFontWeight = Number(process.env.DASHBOARD_PREVIEW_LABEL_FONT_WEIGHT ?? 600);
const labelFontWeight = Number.isFinite(configuredLabelFontWeight)
  ? Math.min(900, Math.max(100, Math.round(configuredLabelFontWeight / 100) * 100))
  : 600;

if (configuredLabelFontFile && !existsSync(configuredLabelFontFile)) {
  throw new Error(`Configured dashboard label font was not found: ${configuredLabelFontFile}`);
}

const fontFiles = [
  sarabunRegularFontFile,
  sarabunSemiBoldFontFile,
  sarabunBoldFontFile,
  ...(configuredLabelFontFile ? [configuredLabelFontFile] : []),
];

GlobalFonts.registerFromPath(sarabunRegularFontFile, "Sarabun");
GlobalFonts.registerFromPath(sarabunSemiBoldFontFile, "Sarabun");
GlobalFonts.registerFromPath(sarabunBoldFontFile, "Sarabun");
if (configuredLabelFontFile) GlobalFonts.registerFromPath(configuredLabelFontFile, labelFontFamily);

interface ChartRow {
  label: string;
  values: number[];
  total: number;
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function compactNumber(value: number): string {
  return Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function capturedLabelSvg(text: string, x: number, y: number, width: number, height: number): string {
  const scale = 2;
  const canvas = createCanvas(width * scale, height * scale);
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#172033";
  context.font = `${labelFontWeight} 18px "${labelFontFamily}"`;
  context.textAlign = "right";
  context.textBaseline = "alphabetic";
  context.fillText(text.normalize("NFC"), width, 20);

  const data = canvas.toBuffer("image/png").toString("base64");
  return `<image x="${x}" y="${y}" width="${width}" height="${height}" href="data:image/png;base64,${data}"/>`;
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}…`;
}

function chartRows(chart: DashboardChart): { rows: ChartRow[]; series: string[] } {
  const series = [...new Set(chart.data.map((datum) => datum.series).filter(Boolean))] as string[];
  const finalSeries = series.length ? series : ["Value"];
  const rowMap = new Map<string, ChartRow>();

  for (const datum of chart.data) {
    const row = rowMap.get(datum.label) ?? {
      label: datum.label,
      values: Array(finalSeries.length).fill(0) as number[],
      total: 0,
    };
    const seriesIndex = series.length ? Math.max(0, finalSeries.indexOf(datum.series ?? "")) : 0;
    const value = datum.value ?? datum.score ?? 0;
    row.values[seriesIndex] += value;
    row.total += value;
    rowMap.set(datum.label, row);
  }

  return { rows: [...rowMap.values()].slice(0, 16), series: finalSeries };
}

function legendSvg(series: string[], top: number): string {
  if (series.length <= 1) return "";
  return series.map((name, index) => {
    const x = 260 + (index % 4) * 220;
    const y = top + Math.floor(index / 4) * 28;
    return [
      `<rect x="${x}" y="${y - 12}" width="14" height="14" rx="3" fill="${COLORS[index % COLORS.length]}"/>`,
      `<text x="${x + 22}" y="${y}" class="legend">${escapeXml(truncate(name, 22))}</text>`,
    ].join("");
  }).join("");
}

function barChartSvg(rows: ChartRow[], series: string[]): { body: string; height: number } {
  const rowHeight = 46;
  const top = series.length > 1 ? 190 + Math.ceil(series.length / 4) * 28 : 190;
  const height = top + rows.length * rowHeight + 70;
  const plotLeft = 310;
  const plotWidth = 780;
  const maxTotal = Math.max(...rows.map((row) => row.total), 1);
  const body: string[] = [legendSvg(series, 155)];

  rows.forEach((row, rowIndex) => {
    const y = top + rowIndex * rowHeight;
    body.push(capturedLabelSvg(truncate(row.label, 32), 0, y - 1, plotLeft - 18, 28));
    body.push(`<line x1="${plotLeft}" y1="${y + 30}" x2="${plotLeft + plotWidth}" y2="${y + 30}" class="grid"/>`);

    let x = plotLeft;
    row.values.forEach((value, seriesIndex) => {
      const width = Math.max(0, value / maxTotal * plotWidth);
      if (width > 0) {
        body.push(
          `<rect x="${x}" y="${y}" width="${width}" height="28" fill="${COLORS[seriesIndex % COLORS.length]}"/>`,
        );
      }
      x += width;
    });
    body.push(`<text x="${Math.min(x + 10, WIDTH - 72)}" y="${y + 19}" class="value">${escapeXml(compactNumber(row.total))}</text>`);
  });

  return { body: body.join(""), height };
}

function lineChartSvg(rows: ChartRow[], series: string[]): { body: string; height: number } {
  const height = 650;
  const plotLeft = 110;
  const plotTop = 205;
  const plotWidth = 980;
  const plotHeight = 340;
  const maxValue = Math.max(...rows.flatMap((row) => row.values), 1);
  const body: string[] = [legendSvg(series, 155)];

  for (let tick = 0; tick <= 4; tick += 1) {
    const y = plotTop + plotHeight - tick / 4 * plotHeight;
    body.push(`<line x1="${plotLeft}" y1="${y}" x2="${plotLeft + plotWidth}" y2="${y}" class="grid"/>`);
    body.push(`<text x="${plotLeft - 14}" y="${y + 5}" text-anchor="end" class="axis">${escapeXml(compactNumber(maxValue * tick / 4))}</text>`);
  }

  series.forEach((_name, seriesIndex) => {
    const points = rows.map((row, rowIndex) => {
      const x = plotLeft + (rows.length <= 1 ? plotWidth / 2 : rowIndex / (rows.length - 1) * plotWidth);
      const y = plotTop + plotHeight - row.values[seriesIndex] / maxValue * plotHeight;
      return `${x},${y}`;
    });
    body.push(`<polyline points="${points.join(" ")}" fill="none" stroke="${COLORS[seriesIndex % COLORS.length]}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>`);
  });

  const labelStep = Math.max(1, Math.ceil(rows.length / 8));
  rows.forEach((row, rowIndex) => {
    if (rowIndex % labelStep !== 0 && rowIndex !== rows.length - 1) return;
    const x = plotLeft + (rows.length <= 1 ? plotWidth / 2 : rowIndex / (rows.length - 1) * plotWidth);
    body.push(`<text x="${x}" y="${plotTop + plotHeight + 30}" text-anchor="middle" class="axis">${escapeXml(truncate(row.label, 14))}</text>`);
  });

  return { body: body.join(""), height };
}

function pieChartSvg(rows: ChartRow[], donut: boolean): { body: string; height: number } {
  const values = rows
    .map((row) => ({ label: row.label, value: Math.max(0, row.values[0] ?? 0) }))
    .filter((row) => row.value > 0)
    .slice(0, 8);
  const total = values.reduce((sum, row) => sum + row.value, 0);
  if (!total) return barChartSvg(rows, [""]);

  const centerX = 330;
  const centerY = 330;
  const radius = 142;
  let angle = -Math.PI / 2;
  const body: string[] = [];
  for (const [index, row] of values.entries()) {
    const nextAngle = angle + (row.value / total) * Math.PI * 2;
    const startX = centerX + Math.cos(angle) * radius;
    const startY = centerY + Math.sin(angle) * radius;
    const endX = centerX + Math.cos(nextAngle) * radius;
    const endY = centerY + Math.sin(nextAngle) * radius;
    const largeArc = nextAngle - angle > Math.PI ? 1 : 0;
    const color = COLORS[index % COLORS.length];
    if (values.length === 1) {
      body.push(`<circle cx="${centerX}" cy="${centerY}" r="${radius}" fill="${color}"/>`);
    } else {
      body.push(
        `<path d="M ${centerX} ${centerY} L ${startX.toFixed(2)} ${startY.toFixed(2)} ` +
        `A ${radius} ${radius} 0 ${largeArc} 1 ${endX.toFixed(2)} ${endY.toFixed(2)} Z" fill="${color}"/>`,
      );
    }
    angle = nextAngle;
  }
  if (donut) {
    body.push(`<circle cx="${centerX}" cy="${centerY}" r="72" fill="#ffffff"/>`);
    body.push(`<text x="${centerX}" y="${centerY + 7}" text-anchor="middle" class="value">${compactNumber(total)}</text>`);
  }
  values.forEach((row, index) => {
    const y = 220 + index * 48;
    const percentage = `${((row.value / total) * 100).toFixed(1)}%`;
    body.push(`<rect x="570" y="${y - 16}" width="18" height="18" rx="4" fill="${COLORS[index % COLORS.length]}"/>`);
    body.push(`<text x="602" y="${y}" class="legend">${escapeXml(truncate(row.label, 32))}</text>`);
    body.push(`<text x="1080" y="${y}" text-anchor="end" class="value">${compactNumber(row.value)} · ${percentage}</text>`);
  });
  return { body: body.join(""), height: Math.max(560, 280 + values.length * 48) };
}

async function renderChartPreview(
  dashboard: DashboardDocument,
  chart: DashboardChart,
): Promise<Buffer | null> {
  if (!chart.data.length) return null;

  const { rows, series } = chartRows(chart);
  if (!rows.length) return null;

  const chartType = chart.chartType.toLowerCase();
  const rendered = chartType.includes("pie") || chartType.includes("donut")
    ? pieChartSvg(rows, chartType.includes("donut"))
    : chartType.includes("line") || chartType.includes("area") || chartType.includes("time")
      ? lineChartSvg(rows, series)
      : barChartSvg(rows, series);

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${rendered.height}" viewBox="0 0 ${WIDTH} ${rendered.height}">
      <style>
        text { font-family: "Sarabun"; fill: #172033; }
        .eyebrow { font-size: 18px; font-weight: 700; letter-spacing: 2px; fill: #2f61df; }
        .title { font-size: 34px; font-weight: 700; }
        .subtitle { font-size: 18px; fill: #586174; }
        .label { font-family: "${escapeXml(labelFontFamily)}"; font-size: 18px; font-weight: ${labelFontWeight}; }
        .legend { font-size: 16px; fill: #586174; }
        .axis { font-size: 15px; fill: #586174; }
        .value { font-size: 16px; font-weight: 600; }
        .grid { stroke: #e7eaf0; stroke-width: 1; }
      </style>
      <rect width="100%" height="100%" fill="#ffffff"/>
      <text x="56" y="48" class="eyebrow">DASHBOARD AGENT</text>
      <text x="56" y="92" class="title">${escapeXml(truncate(chart.title || dashboard.title, 64))}</text>
      <text x="56" y="126" class="subtitle">${escapeXml(truncate(chart.reason ?? dashboard.subtitle ?? dashboard.query, 110))}</text>
      ${rendered.body}
      <text x="56" y="${rendered.height - 24}" class="axis">Static preview • ${escapeXml(dashboard.execution.source)} • ${escapeXml(new Date(dashboard.generatedAt).toLocaleString("en"))}</text>
    </svg>
  `;

  const renderer = new Resvg(svg, {
    font: {
      fontFiles,
      defaultFontFamily: "Sarabun",
      loadSystemFonts: false,
    },
  });
  return Buffer.from(renderer.render().asPng());
}

export async function renderDashboardPreview(dashboard: DashboardDocument): Promise<Buffer | null> {
  const chart = dashboard.charts[0];
  return chart ? renderChartPreview(dashboard, chart) : null;
}

export async function renderDashboardPreviews(dashboard: DashboardDocument): Promise<Buffer[]> {
  const previews = await Promise.all(
    dashboard.charts.map((chart) => renderChartPreview(dashboard, chart)),
  );
  return previews.filter((preview): preview is Buffer => preview !== null);
}
