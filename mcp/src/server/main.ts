import "dotenv/config";

import { createServer } from "node:http";
import path from "node:path";

import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { Request, Response } from "express";

import { createDashboardPreviewStore } from "./dashboard-preview-store.js";
import { createDashboardMcpServer } from "./server.js";

async function runStdio(): Promise<void> {
  const server = createDashboardMcpServer();
  await server.connect(new StdioServerTransport());
}

async function runHttp(): Promise<void> {
  const host = process.env.MCP_HOST ?? "127.0.0.1";
  const port = Number(process.env.MCP_PORT ?? 3001);
  const publicBaseUrl = (process.env.MCP_PUBLIC_BASE_URL ?? `http://127.0.0.1:${port}`)
    .replace(/\/+$/, "");
  const previewTtlMs = Number(process.env.DASHBOARD_PREVIEW_TTL_MS ?? 7 * 86_400_000);
  const previewDirectory =
    process.env.DASHBOARD_PREVIEW_DIR ?? path.resolve(process.cwd(), "data", "previews");
  const previews = createDashboardPreviewStore({
    directory: previewDirectory,
    ttlMs: previewTtlMs,
  });
  const allowedHosts = process.env.MCP_ALLOWED_HOSTS
    ?.split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const app = createMcpExpressApp({
    host,
    ...(allowedHosts?.length ? { allowedHosts } : {}),
  });

  app.get("/health", (_request: Request, response: Response) => {
    response.json({ ok: true, service: "dashboard-agent-mcp" });
  });

  app.get("/dashboard-previews/:id.png", async (request: Request, response: Response) => {
    const id = Array.isArray(request.params.id) ? request.params.id[0] : request.params.id;
    let preview: Buffer | undefined;
    try {
      preview = id ? await previews.get(id) : undefined;
    } catch {
      preview = undefined;
    }
    if (!preview) {
      response.status(404).json({ error: "Dashboard preview not found or expired" });
      return;
    }
    response
      .set({
        "Cache-Control": `private, max-age=${Math.max(0, Math.floor(previewTtlMs / 1000))}`,
        "Content-Type": "image/png",
        "X-Content-Type-Options": "nosniff",
      })
      .send(preview);
  });

  app.all("/mcp", async (request: Request, response: Response) => {
    const server = createDashboardMcpServer({
      publishPreview: async (preview) => {
        const id = await previews.put(preview);
        return `${publicBaseUrl}/dashboard-previews/${id}.png`;
      },
    });
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    response.on("close", () => {
      void transport.close();
      void server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(request, response, request.body);
  });

  createServer(app).listen(port, host, () => {
    process.stderr.write(`Dashboard Agent MCP listening on http://${host}:${port}/mcp\n`);
  });
}

if (process.argv.includes("--stdio")) {
  await runStdio();
} else {
  await runHttp();
}
