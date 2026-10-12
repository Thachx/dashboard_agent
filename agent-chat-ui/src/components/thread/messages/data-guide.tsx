"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  Database,
  Fingerprint,
  Hash,
  Search,
  Table2,
  Timer,
} from "lucide-react";

type GuideColumn = {
  name: string;
  dataType: string;
  isTime: boolean;
  isIdentifier: boolean;
  terms: string[];
  samples: string[];
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

type GuideSourceTable = {
  name: string;
  sourceSystem: string;
  sourcePath: string;
  rowCount: number;
  ingestionMode: string;
  ingestedAt: string | null;
  columns: { name: string; dataType: string; thaiDescription?: string | null }[];
  thaiSummary?: string | null;
};

type GuidePayload = {
  version: number;
  warehouse: string;
  tableCount: number;
  tables: GuideTable[];
  sourceTables?: GuideSourceTable[];
  thaiOverview?: string | null;
  error?: string;
};

// Thai labels use the same codepoints as the rest of the widget
// (house spelling), written as \u escapes so the source stays ASCII.
const THAI = {
  title: "คู่มือตารางและฟิลด์",
  overview: "เกี่ยวกับข้อมูลนี้",
  sourceTables: "ตารางต้นฉบับ",
  search: "ค้นหาฟิลด์...",
  tables: "ตารางข้อมูล",
  rows: "แถว",
  columns: "คอลัมน์",
  samples: "ตัวอย่างค่า",
  used: "ใช้ในแดชบอร์ดนี้",
  ask: "ถามเกี่ยวกับฟิลด์นี้",
  loading: "กำลังโหลดคู่มือข้อมูล...",
  error: "ไม่สามารถโหลดคู่มือข้อมูลได้",
  noMatch: "ไม่พบฟิลด์ที่ตรงกัน",
  time: "วันที่/เวลา",
  identifier: "ตัวระบุ",
};

function humanizeField(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
}

function askAboutField(column: GuideColumn): string {
  const label = humanizeField(column.name);
  return column.isTime
    ? `Show ${label} over time`
    : `Show the breakdown by ${label}`;
}

export function DataGuide({
  usedFields,
  language = "en",
}: {
  usedFields?: string[];
  language?: string;
}) {
  const isThai = language === "th";
  const t = isThai ? THAI : null;
  const [open, setOpen] = useState(false);
  const [payload, setPayload] = useState<GuidePayload | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

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

  const used = useMemo(() => new Set(usedFields ?? []), [usedFields]);

  const filteredTables = useMemo(() => {
    if (!payload) return [];
    const q = query.trim().toLowerCase();
    // Merge the original source tables into the same list as the derived
    // tables, so they appear as regular (badged) entries in one guide.
    const allTables: GuideTable[] = [
      ...payload.tables,
      ...(payload.sourceTables ?? []).map((st) => {
        const tableShort =
          (st.sourcePath ? st.sourcePath.split("/").pop() : undefined) ??
          st.name;
        const sourceLabel = st.sourceSystem
          ? `${st.sourceSystem}/${tableShort}`
          : tableShort;
        return {
          name: st.name,
          rowCount: st.rowCount,
          sourcePaths: [sourceLabel],
          columns: st.columns.map((c) => ({
            name: c.name,
            dataType: c.dataType,
            isTime: false,
            isIdentifier: false,
            terms: [] as string[],
            samples: [] as string[],
          })),
          thaiSummary: st.thaiSummary,
          isSource: true,
          sourceSystem: st.sourceSystem,
          sourceLabel: sourceLabel,
        };
      }),
    ];
    if (!q) return allTables;
    return allTables
      .map((table) => ({
        table,
        tableMatch: table.name.toLowerCase().includes(q),
        columns: table.columns.filter(
          (col) =>
            col.name.toLowerCase().includes(q) ||
            col.dataType.toLowerCase().includes(q) ||
            col.terms.some((term) => term.toLowerCase().includes(q)),
        ),
      }))
      .map((entry) => ({
        ...entry.table,
        columns: entry.tableMatch ? entry.table.columns : entry.columns,
      }))
      .filter((table) => table.columns.length > 0);
  }, [payload, query]);

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="group inline-flex max-w-full items-center gap-1 text-left text-xs text-foreground/35 transition hover:text-foreground/70"
        aria-expanded={open}
      >
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
        <Table2 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <span className="truncate">
          {t ? t.title : "Data guide (tables & fields)"}
        </span>
      </button>
      {open ? (
        <div className="mt-2 rounded-lg border bg-muted/10 p-3 text-xs">
          {payload?.error ? (
            <p className="text-muted-foreground">
              {t ? t.error : "Could not load the data guide."}
            </p>
          ) : !payload ? (
            <p className="text-muted-foreground">
              {t ? t.loading : "Loading data guide..."}
            </p>
          ) : (
            <>
              {payload.thaiOverview ? (
                <div className="mt-2 rounded-lg border bg-muted/20 px-3 py-2">
                  <p className="text-xs font-semibold">{t ? t.overview : "About this data"}</p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{payload.thaiOverview}</p>
                </div>
              ) : null}

              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-medium text-foreground/80">
                  {t ? t.tables : "Tables"} ({payload.tableCount})
                </p>
                <label className="relative flex-1 sm:max-w-[240px]">
                  <Search
                    className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
                    aria-hidden="true"
                  />
                  <input
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder={t ? t.search : "Search fields..."}
                    className="w-full rounded-md border bg-background py-1 pl-7 pr-2 text-xs outline-none transition focus:border-foreground/30"
                  />
                </label>
              </div>
              {filteredTables.length === 0 ? (
                <p className="mt-3 text-muted-foreground">
                  {t ? t.noMatch : "No fields match your search."}
                </p>
              ) : (
                <div className="mt-3 space-y-2">
                  {filteredTables.map((table) => {
                    const isOpen = expanded === table.name;
                    return (
                      <div
                        key={table.name}
                        className="rounded-md border bg-background"
                      >
                        <button
                          type="button"
                          onClick={() =>
                            setExpanded(isOpen ? null : table.name)
                          }
                          className="flex w-full items-center gap-2 px-3 py-2 text-left"
                        >
                          <Database
                            className="h-3.5 w-3.5 shrink-0 text-muted-foreground"
                            aria-hidden="true"
                          />
                          <span className="min-w-0 flex-1 truncate font-medium">
                            {table.sourceLabel ?? humanizeField(table.name)}
                          </span>
                          <span className="shrink-0 text-muted-foreground">
                            {table.rowCount.toLocaleString()}{" "}
                            {t ? t.rows : "rows"}
                          </span>
                          <span className="shrink-0 text-muted-foreground">
                            {table.columns.length}{" "}
                            {t ? t.columns : "cols"}
                          </span>
                          <ChevronDown
                            className={`h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
                            aria-hidden="true"
                          />
                        </button>
                        {isOpen ? (
                          <div className="border-t px-3 py-2">
                            {table.sourcePaths.length ? (
                              <p
                                className="mb-2 truncate text-muted-foreground"
                                title={table.sourcePaths.join(", ")}
                              >
                                {table.sourcePaths.join(", ")}
                              </p>
                            ) : null}
                            <div className="grid gap-1">
                              {table.columns.map((column) => (
                                <div
                                  key={column.name}
                                  className="grid gap-1 rounded-md bg-muted/20 px-2 py-1.5 sm:grid-cols-[minmax(0,1.2fr)_auto_auto_minmax(0,1fr)_auto]"
                                >
                                  <div className="flex min-w-0 items-center gap-1.5">
                                    <span
                                      className="truncate font-medium"
                                      title={column.name}
                                    >
                                      {humanizeField(column.name)}
                                    </span>
                                    {column.isTime ? (
                                      <Timer
                                        className="h-3 w-3 shrink-0 text-muted-foreground"
                                        aria-label={t ? t.time : "time field"}
                                      />
                                    ) : null}
                                    {column.isIdentifier ? (
                                      <Fingerprint
                                        className="h-3 w-3 shrink-0 text-muted-foreground"
                                        aria-label={t
                                          ? t.identifier
                                          : "identifier"}
                                      />
                                    ) : null}
                                  </div>
                                  <span
                                    className="flex items-center gap-1 text-muted-foreground"
                                    title={column.dataType}
                                  >
                                    <Hash
                                      className="h-3 w-3 shrink-0"
                                      aria-hidden="true"
                                    />
                                    <span className="max-w-[90px] truncate">
                                      {column.dataType}
                                    </span>
                                  </span>
                                  {used.has(column.name) ? (
                                    <span className="justify-self-start rounded border border-emerald-600/30 bg-emerald-600/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700">
                                      {t ? t.used : "used here"}
                                    </span>
                                  ) : (
                                    <span />
                                  )}
                                  <span
                                    className="min-w-0 truncate text-muted-foreground"
                                    title={column.samples.join(", ")}
                                  >
                                    {column.samples.length
                                      ? column.samples.join(", ")
                                      : "\u2014"}
                                  </span>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      window.dispatchEvent(
                                        new CustomEvent("dashboard:prefill-question", {
                                          detail: askAboutField(column),
                                        }),
                                      )
                                    }
                                    className="justify-self-end rounded border bg-background px-2 py-0.5 text-[10px] font-medium text-foreground/70 transition hover:border-foreground/30 hover:text-foreground"
                                  >
                                    {t ? t.ask : "Ask about this field"}
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
