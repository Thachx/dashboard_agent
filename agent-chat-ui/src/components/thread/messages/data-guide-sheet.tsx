"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  Fingerprint,
  Hash,
  MessageSquareText,
  Search,
  Timer,
} from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

type GuideColumn = {
  name: string;
  dataType: string;
  isTime: boolean;
  isIdentifier: boolean;
  terms: string[];
  samples: string[];
  thaiDescription?: string | null;
};
type GuideTable = {
  name: string;
  rowCount: number;
  sourcePaths: string[];
  columns: GuideColumn[];
  thaiSummary?: string | null;
  isSource?: boolean;
  sourceSystem?: string;
  sourceLabel?: string;
};
type GuideSourceColumn = {
  name: string;
  dataType: string;
  thaiDescription?: string | null;
};

type GuideSourceTable = {
  name: string;
  sourceSystem: string;
  sourcePath: string;
  rowCount: number;
  ingestionMode: string;
  ingestedAt: string | null;
  columns: GuideSourceColumn[];
  thaiSummary?: string | null;
};

type GuidePayload = {
  version: number;
  warehouse: string;
  tableCount: number;
  tables: GuideTable[];
  sourceTables?: GuideSourceTable[];
  thaiStatus?: string;
  thaiOverview?: string | null;
  error?: string;
};

function humanizeField(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
}

// Shorten a source path for display: strip the s3://<bucket>/ prefix, then
// show "top folder / last segment" (e.g. s3://lead-etl/profile/raw/mysql/fct_profile_institute
// -> profile/fct_profile_institute). Paths with 2 or fewer segments stay as-is.
function shortSourcePath(p: string): string {
  const noScheme = p.replace(/^s3:\/\/[^/]+\//, "");
  const parts = noScheme.split("/").filter(Boolean);
  if (parts.length <= 2) return noScheme;
  return `${parts[0]}/${parts[parts.length - 1]}`;
}


function describeField(col: GuideColumn, tableName: string): string {
  const dt = col.dataType.toUpperCase();
  let role: string;
  if (col.isTime) role = "วันที่/เวลา";
  else if (col.isIdentifier) role = "ตัวระบุ";
  else if (/(BIGINT|INT|DOUBLE|FLOAT|DECIMAL|NUMERIC|REAL|SMALLINT|TINYINT|HUGEINT)/.test(dt))
    role = "ตัวเลข";
  else if (/BOOLEAN/.test(dt)) role = "ใช่/ไม่ใช่";
  else if (/VARCHAR|TEXT|STRING|CHAR/.test(dt)) role = "ข้อความ";
  else role = dt;

  const parts = [`${role} ฟิลด์`, `ใน ${humanizeField(tableName)}`];
  if (col.samples.length) parts.push(`เช่น ${col.samples.slice(0, 3).join(", ")}`);
  return parts.join(" \u00b7 ");
}


function askAboutField(col: GuideColumn): string {
  const label = humanizeField(col.name);
  return col.isTime ? `แสดง {label} ตามเวลา ${label}` : `แสดงการจำแนกตาม {label} ${label}`;
}


export function DataGuideSheet() {
  const [open, setOpen] = useState(false);
  const [payload, setPayload] = useState<GuidePayload | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open || payload || failed) return;
    let cancelled = false;
    fetch("/api/catalog")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setPayload(data);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [open, payload, failed]);

  const filteredTables = useMemo(() => {
    if (!payload) return [];
    const q = query.trim().toLowerCase();
    const out: { table: GuideTable; cols: GuideColumn[] }[] = [];
    // Merge the original source tables into the same list as the derived
    // tables, so they appear as regular (badged) entries in one guide.
    const allTables: GuideTable[] = [
      ...payload.tables,
      ...(payload.sourceTables ?? []).map((st) => ({
        name: st.name,
        rowCount: st.rowCount,
        sourcePaths: [st.sourcePath],
        sourceLabel: `${st.sourceSystem}/${st.sourcePath.split("/").pop() ?? st.name}`,
        columns: st.columns.map((c) => ({
          name: c.name,
          dataType: c.dataType,
          isTime: false,
          isIdentifier: false,
          terms: [] as string[],
          samples: [] as string[],
          thaiDescription: c.thaiDescription,
        })),
        thaiSummary: st.thaiSummary,
        isSource: true,
        sourceSystem: st.sourceSystem,
      })),
    ];
    for (const table of allTables) {
      const tableHay = [
        table.name,
        humanizeField(table.name),
        table.thaiSummary ?? "",
      ]
        .join(" ")
        .toLowerCase();
      const cols = table.columns.filter((col) => {
        if (!q) return true;
        const hay = [
          col.name,
          humanizeField(col.name),
          col.dataType,
          col.thaiDescription ?? "",
          table.name,
          humanizeField(table.name),
          ...col.terms,
          ...col.samples,
        ]
          .join(" ")
          .toLowerCase();
        return hay.includes(q) || tableHay.includes(q);
      });
      if (cols.length) out.push({ table, cols });
    }
    return out;
  }, [payload, query]);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground/80 shadow-xs transition hover:border-foreground/30 hover:text-foreground"
        >
          <BookOpen className="h-4 w-4" aria-hidden="true" />
          Data guide
        </button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 p-0 sm:max-w-[440px]"
      >
        <SheetHeader className="border-b bg-muted/30 px-4 py-3">
          <SheetTitle className="flex items-center gap-2 text-base">
            <BookOpen className="h-4 w-4" aria-hidden="true" />
            Data guide
          </SheetTitle>
          <SheetDescription className="text-xs leading-loose">
            ค้นหาตารางและฟิลด์ที่คุณสามารถถามได้
          </SheetDescription>
        </SheetHeader>

        {payload?.thaiOverview ? (
          <div className="rounded-lg border bg-muted/20 px-3 py-2">
            <p className="text-xs font-semibold">เกี่ยวกับข้อมูลนี้</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{payload?.thaiOverview}</p>
          </div>
        ) : null}

        <div className="border-b px-4 py-2">
          <label className="relative block">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <input
              type="search"
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="ค้นหาฟิลด์ ตาราง หรือค่า..."
              className="w-full rounded-md border bg-background py-3 pl-8 pr-3 text-sm leading-normal outline-none transition focus:border-foreground/30"
            />
          </label>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3">
          {payload?.error ? (
            <p className="px-1 text-sm text-muted-foreground">ไม่สามารถโหลดคู่มือข้อมูลได้</p>
          ) : !payload ? (
            <p className="px-1 text-sm text-muted-foreground">กำลังโหลดคู่มือข้อมูล...</p>
          ) : filteredTables.length === 0 ? (
            <p className="px-1 text-sm text-muted-foreground">
              ไม่มีฟิลด์ที่ตรงกัน "{query}".
            </p>
          ) : (
            <div className="space-y-4">
              {payload.thaiStatus === "pending" ? (
                <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
                  กำลังสร้างคำอธิบายฟิลด์เป็นภาษาไทย......
                </p>
              ) : null}
              {filteredTables.map(({table, cols}) => (
                <section key={table.name} className="space-y-2">
                  <div className="px-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold">
                        {table.isSource && table.sourceLabel
                          ? table.sourceLabel
                          : humanizeField(table.name)}
                      </h3>
                      {table.isSource ? (
                        <span className="shrink-0 rounded border border-foreground/20 bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                          {table.sourceSystem}
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {table.thaiSummary ||
                        `${table.rowCount.toLocaleString()} แถว`}
                    </p>
                    {!table.isSource && table.sourcePaths.length ? (
                      <p
                        className="mt-0.5 truncate text-[11px] text-muted-foreground/70"
                        title={table.sourcePaths.join(", ")}
                      >
                        {table.sourcePaths.map(shortSourcePath).join(", ")}
                      </p>
                    ) : null}
                  </div>
                  <div className="space-y-2">
                    {cols.map((col) => (
                      <div
                        key={`${table.name}.${col.name}`}
                        className="rounded-lg border bg-background p-3"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex min-w-0 items-center gap-1.5">
                            <span className="truncate text-sm font-semibold">
                              {humanizeField(col.name)}
                            </span>
                            {col.isTime ? (
                              <Timer
                                className="h-3.5 w-3.5 shrink-0 text-muted-foreground"
                                aria-label="time field"
                              />
                            ) : null}
                            {col.isIdentifier ? (
                              <Fingerprint
                                className="h-3.5 w-3.5 shrink-0 text-muted-foreground"
                                aria-label="identifier"
                              />
                            ) : null}
                          </div>
                          <span
                            className="flex shrink-0 items-center gap-1 rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
                            title={col.dataType}
                          >
                            <Hash className="h-3 w-3" aria-hidden="true" />
                            {col.dataType}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {col.thaiDescription || describeField(col, table.name)}
                        </p>
                        {col.samples.length ? (
                          <p className="mt-1 text-xs text-muted-foreground/80">
                            เช่น {col.samples.slice(0, 3).join(", ")}
                          </p>
                        ) : null}
                        <button
                          type="button"
                          onClick={() => {
                            window.dispatchEvent(
                              new CustomEvent("dashboard:prefill-question", {
                                detail: askAboutField(col),
                              }),
                            );
                            setOpen(false);
                          }}
                          className="mt-2 inline-flex items-center gap-1.5 rounded-md border bg-background px-2.5 py-1 text-xs font-medium text-foreground/70 transition hover:border-foreground/30 hover:text-foreground"
                        >
                          <MessageSquareText className="h-3.5 w-3.5" aria-hidden="true" />
                          ถามเกี่ยวกับฟิลด์นี้
                        </button>
                      </div>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
