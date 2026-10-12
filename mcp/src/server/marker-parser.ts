const START = "<<<GRAPH_DASHBOARD_WIDGET>>>";
const END = "<<<END_GRAPH_DASHBOARD_WIDGET>>>";

type UnknownRecord = Record<string, unknown>;

function record(value: unknown): UnknownRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : {};
}

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      const value = record(part);
      return typeof value.text === "string" ? value.text : "";
    })
    .filter(Boolean)
    .join("\n");
}

export function assistantTextFromLangGraph(value: unknown): string {
  if (typeof value === "string") return value;
  const root = record(value);
  const containers = [root.messages, record(root.values).messages, record(root.output).messages];
  for (const messages of containers) {
    if (!Array.isArray(messages)) continue;
    for (const message of [...messages].reverse()) {
      const item = record(message);
      const role = String(item.role ?? item.type ?? "").toLowerCase();
      if (role && !role.includes("assistant") && !role.includes("ai")) continue;
      const text = contentText(item.content);
      if (text) return text;
    }
  }
  return contentText(root.content);
}

export function tryParseGraphDashboardMarker(text: string): unknown | undefined {
  const payloads: string[] = [];
  let offset = 0;
  while (offset < text.length) {
    const start = text.indexOf(START, offset);
    if (start < 0) break;
    const bodyStart = start + START.length;
    const end = text.indexOf(END, bodyStart);
    if (end < 0) break;
    payloads.push(text.slice(bodyStart, end).trim());
    offset = end + END.length;
  }
  const latest = payloads.at(-1);
  if (!latest) {
    if (text.includes(START)) {
      throw new Error("Dashboard Agent returned an incomplete graph dashboard widget payload.");
    }
    return undefined;
  }
  try {
    return JSON.parse(latest) as unknown;
  } catch (error) {
    throw new Error(`Dashboard widget payload is invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export function parseGraphDashboardMarker(text: string): unknown {
  const payload = tryParseGraphDashboardMarker(text);
  if (payload === undefined) {
    throw new Error("Dashboard Agent returned no graph dashboard widget payload.");
  }
  return payload;
}
