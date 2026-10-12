import { describe, expect, it } from "vitest";

import { dashboardFromTextResponse } from "../src/shared/dashboard-contract";

describe("dashboard contract adapter", () => {
  it("normalizes non-chart analytical responses without inventing a chart", () => {
    const dashboard = dashboardFromTextResponse(
      "## User count\nThere are **42 active users**.",
      { query: "How many active users are there?", locale: "en", timezone: "Asia/Bangkok" },
    );

    expect(dashboard).toMatchObject({
      title: "User count",
      summary: "There are **42 active users**.",
      query: "How many active users are there?",
      locale: "en",
      timezone: "Asia/Bangkok",
      metrics: [],
      charts: [],
      layout: [],
    });
  });
});
