import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import { createDashboard } from "./dashboard-adapter.js";
import { renderDashboardPreviews } from "./dashboard-image.js";

export const DASHBOARD_RESOURCE_URI = "ui://dashboard-agent/dashboard-app.html";
export const DASHBOARD_TOOL_NAME = "create_dashboard";

const resourcePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../ui/dashboard-app.html",
);

export function createDashboardMcpServer(options: {
  resourceHtml?: string;
  publishPreview?: (preview: Buffer) => string | Promise<string>;
} = {}): McpServer {
  const server = new McpServer({
    name: "dashboard-agent-mcp",
    version: "0.1.0",
  });

  registerAppTool(
    server,
    DASHBOARD_TOOL_NAME,
    {
      title: "Analyze data or create a dashboard",
      description:
        "Answer a source-backed analytical question and generate charts only when the request calls for them. " +
        "Use this for metrics, factual data questions, record lookups, comparisons, distributions, trends, or joined analysis. " +
        "Always use this tool when the user's message starts with /dashboard, passing the text after the command " +
        "as the question. Chart requests include one Markdown image per generated chart; non-chart requests return " +
        "the analytical answer as text. After a successful call, return the tool's text content verbatim as the final response. " +
        "Do not omit, rewrite, reorder, or repeat its image lines, and do not add substitute analysis.",
      inputSchema: {
        question: z.string().min(2).describe("The user's analytical question."),
        locale: z.string().optional().describe("BCP 47 response locale, such as en or th."),
        timezone: z.string().optional().describe("IANA timezone, such as Asia/Bangkok."),
        maxCharts: z.number().int().min(1).max(8).optional(),
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: true,
      },
      _meta: {
        ui: { resourceUri: DASHBOARD_RESOURCE_URI },
      },
    },
    async (request) => {
      try {
        const dashboard = await createDashboard(request);
        const previews = await renderDashboardPreviews(dashboard);
        const previewUrls = options.publishPreview
          ? await Promise.all(previews.map((preview) => options.publishPreview!(preview)))
          : [];
        const imageLines = previewUrls.map((url, index) => {
          const chartTitle = dashboard.charts[index]?.title ?? `Dashboard chart ${index + 1}`;
          const markdownAlt = chartTitle.replaceAll(/[[\]\r\n]/g, "");
          return `![${markdownAlt}](${url})`;
        });
        const responseText = [
          `### ${dashboard.title}`,
          dashboard.summary,
          ...(imageLines.length ? ["", ...imageLines] : []),
        ].join("\n");
        // Native MCP image blocks let hosts such as Open WebUI attach the
        // rendered charts directly to the chat message.
        const imageContent = previews.slice(0, 6).map((png) => ({
          type: "image" as const,
          data: png.toString("base64"),
          mimeType: "image/png",
        }));
        return {
          content: [{ type: "text", text: responseText }, ...imageContent],
          structuredContent: dashboard,
          ...(previewUrls.length
            ? { _meta: { previewUrl: previewUrls[0], previewUrls } }
            : {}),
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown dashboard error";
        return {
          isError: true,
          content: [{ type: "text", text: `Could not create the dashboard: ${message}` }],
        };
      }
    },
  );

  registerAppResource(
    server,
    "Dashboard Agent view",
    DASHBOARD_RESOURCE_URI,
    {
      description: "Interactive renderer for Dashboard Agent results.",
      mimeType: RESOURCE_MIME_TYPE,
    },
    async () => ({
      contents: [
        {
          uri: DASHBOARD_RESOURCE_URI,
          mimeType: RESOURCE_MIME_TYPE,
          text: options.resourceHtml ?? await readFile(resourcePath, "utf8"),
          _meta: {
            ui: {
              csp: { resourceDomains: [], connectDomains: [] },
            },
          },
        },
      ],
    }),
  );

  return server;
}
