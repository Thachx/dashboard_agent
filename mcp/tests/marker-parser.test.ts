import { describe, expect, it } from "vitest";

import {
  assistantTextFromLangGraph,
  parseGraphDashboardMarker,
  tryParseGraphDashboardMarker,
} from "../src/server/marker-parser";

describe("dashboard marker parser", () => {
  it("extracts the latest graph dashboard payload", () => {
    const text = [
      "Older result",
      '<<<GRAPH_DASHBOARD_WIDGET>>>{"title":"Old"}<<<END_GRAPH_DASHBOARD_WIDGET>>>' ,
      '<<<GRAPH_DASHBOARD_WIDGET>>>{"title":"Current","activity":{"chartSlots":[]}}<<<END_GRAPH_DASHBOARD_WIDGET>>>',
    ].join("\n");
    expect(parseGraphDashboardMarker(text)).toMatchObject({ title: "Current" });
  });

  it("reads the latest assistant message from LangGraph state", () => {
    const response = {
      values: {
        messages: [
          { type: "human", content: "question" },
          { type: "ai", content: [{ type: "text", text: "dashboard result" }] },
        ],
      },
    };
    expect(assistantTextFromLangGraph(response)).toBe("dashboard result");
  });

  it("allows a plain non-chart assistant response", () => {
    expect(tryParseGraphDashboardMarker("There are 42 active users.")).toBeUndefined();
    expect(() => parseGraphDashboardMarker("There are 42 active users."))
      .toThrow("no graph dashboard widget payload");
  });

  it("still rejects a truncated dashboard marker", () => {
    expect(() => tryParseGraphDashboardMarker('<<<GRAPH_DASHBOARD_WIDGET>>>{"title":"Partial"}'))
      .toThrow("incomplete graph dashboard widget payload");
  });
});
