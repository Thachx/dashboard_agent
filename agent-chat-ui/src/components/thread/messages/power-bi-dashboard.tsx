"use client";

import { useEffect, useRef } from "react";

export type PowerBiDashboardPayload = {
  version: number;
  kind: "power-bi-dashboard";
  title: string;
  embedUrl: string;
  reportId?: string | null;
  accessToken?: string | null;
};

export function PowerBiDashboard({ payload }: { payload: PowerBiDashboardPayload }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !payload.accessToken) return;
    const container = containerRef.current;
    const accessToken = payload.accessToken;
    let active = true;
    let reset: (() => void) | undefined;
    void import("powerbi-client").then(({ models, service, factories }) => {
      if (!active) return;
      const powerbi = new service.Service(
        factories.hpmFactory,
        factories.wpmpFactory,
        factories.routerFactory,
      );
      powerbi.embed(container, {
        type: "report",
        id: payload.reportId ?? undefined,
        embedUrl: payload.embedUrl,
        accessToken,
        tokenType: models.TokenType.Embed,
        settings: {
          panes: { filters: { visible: false }, pageNavigation: { visible: true } },
          background: models.BackgroundType.Transparent,
        },
      });
      reset = () => powerbi.reset(container);
    });
    return () => {
      active = false;
      reset?.();
    };
  }, [payload.accessToken, payload.embedUrl, payload.reportId]);

  return (
    <section className="mt-3 overflow-hidden rounded-xl border border-border bg-background shadow-sm">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <p className="text-sm font-semibold">{payload.title}</p>
          <p className="text-xs text-muted-foreground">Interactive Power BI report</p>
        </div>
        <a href={payload.embedUrl} target="_blank" rel="noreferrer" className="text-xs font-medium text-blue-600 hover:underline">
          Open report
        </a>
      </header>
      {payload.accessToken ? (
        <div ref={containerRef} className="h-[620px] w-full bg-muted/10" />
      ) : (
        <iframe title={payload.title} src={payload.embedUrl} className="h-[620px] w-full border-0" allowFullScreen />
      )}
    </section>
  );
}
