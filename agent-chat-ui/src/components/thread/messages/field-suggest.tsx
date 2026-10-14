"use client";
// Field autosuggest for the chat composer.
//
// General, data-driven: it fetches the field catalog once, then matches the
// user's ACTIVE TYPED TOKEN against every field's name, English terms, and
// Thai description. It works for English ("grade", "user") and Thai
// ("คะแนน") alike because the token is just the trailing run of word
// characters (ASCII or Thai script) — no per-field or per-language
// hardcoding. Picking a field inserts its readable name so the question
// names the field explicitly.
//
// Thai insertion is SURGICAL: Thai has no word boundaries, so the typed
// token is a whole run ("แสดงคะแนนตามรายวิชา"). Matching reports the span of
// the run that actually matched a field ("คะแนน"), and only that span is
// replaced — the rest of the sentence is left untouched. English tokens are
// single words, so their span is the whole token (unchanged behavior).
// Matching is tone-mark-insensitive (see field-suggest-match.ts).
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Clock, Fingerprint } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  humanizeField,
  scoreFieldWithSpan,
  trailingToken,
  type MatchSpan,
} from "./field-suggest-match";

type SuggestColumn = {
  name: string;
  dataType: string;
  isTime: boolean;
  isIdentifier: boolean;
  terms: string[];
  samples: string[];
  thaiDescription?: string | null;
};
type SuggestTable = {
  name: string;
  columns: SuggestColumn[];
  sourceSystem?: string;
  sourcePath?: string;
};
type SuggestPayload = {
  tables: SuggestTable[];
  sourceTables?: SuggestTable[];
};

type Suggestion = {
  key: string;
  field: string; // canonical column name
  label: string; // readable (humanized) name
  table: string;
  tableLabel?: string; // parent/table for source tables
  isTime: boolean;
  isIdentifier: boolean;
  thaiDescription: string;
  score: number;
  span: MatchSpan; // part of the typed token this field matched
};

// House spelling (same codepoints as the data guide) for the panel header.
const T_FIELD = "\u0e1f\u0e34\u0e25\u0e14\u0e4c"; // field

function buildSuggestions(
  payload: SuggestPayload | null,
  token: string,
): Suggestion[] {
  if (!payload || token.length < 2) return [];
  const seen = new Set<string>();
  const out: Suggestion[] = [];
  const allTables = [
    ...(payload.tables ?? []),
    ...(payload.sourceTables ?? []),
  ];
  for (const table of allTables) {
    const tableLabel = table.sourceSystem
      ? `${table.sourceSystem}/${(table.sourcePath ?? table.name).split("/").pop() ?? table.name}`
      : undefined;
    for (const col of table.columns ?? []) {
      if (seen.has(col.name)) continue;
      seen.add(col.name);
      const { score, span } = scoreFieldWithSpan(col, token);
      if (score <= 0 || !span) continue;
      out.push({
        key: `${table.name}.${col.name}`,
        field: col.name,
        label: humanizeField(col.name),
        table: table.name,
        tableLabel,
        isTime: !!col.isTime,
        isIdentifier: !!col.isIdentifier,
        thaiDescription: col.thaiDescription ?? "",
        score,
        span,
      });
    }
  }
  out.sort((a, b) => b.score - a.score || a.label.localeCompare(b.label));
  return out.slice(0, 6);
}

export function useFieldSuggest(opts: {
  input: string;
  setInput: (v: string) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const { input, setInput, textareaRef } = opts;
  const [payload, setPayload] = useState<SuggestPayload | null>(null);
  const [dismissed, setDismissed] = useState(false);

  // Fetch the catalog once.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/catalog")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setPayload(data as SuggestPayload);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const tokenInfo = useMemo(() => trailingToken(input), [input]);
  const token = tokenInfo?.token ?? "";

  // A new/changed token re-arms the dropdown (clears a prior Escape-dismiss).
  useEffect(() => {
    setDismissed(false);
  }, [token]);

  const suggestions = useMemo(
    () => buildSuggestions(payload, token),
    [payload, token],
  );
  const visible = !dismissed && suggestions.length > 0 && !!tokenInfo;

  const [index, setIndex] = useState(0);
  useEffect(() => {
    setIndex(0);
  }, [token]);

  const pick = useCallback(
    (s: Suggestion) => {
      const info = trailingToken(input);
      if (!info) return;
      // Replace only the matched span of the typed token, not the whole run:
      // "แสดงคะแนนตามรายวิชา" + pick(คะแนน) -> "แสดง Grade ตามรายวิชา".
      const absStart = info.start + s.span.start;
      const absEnd = info.start + s.span.end;
      const prefix = input.slice(0, absStart);
      const suffix = input.slice(absEnd);
      const needsSpace = suffix.length > 0 && !/^\s/.test(suffix);
      const next = prefix + s.label + (needsSpace ? " " : "") + suffix;
      setInput(next);
      setDismissed(true);
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        const pos = absStart + s.label.length + (needsSpace ? 1 : 0);
        el.focus();
        el.setSelectionRange(pos, pos);
      });
    },
    [input, setInput, textareaRef],
  );

  // Returns true when the key was consumed by the suggest UI, so the caller
  // skips its own handling (e.g. Enter-to-submit).
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>): boolean => {
      if (!visible) return false;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setIndex((i) => (i + 1) % suggestions.length);
        return true;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
        return true;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        pick(suggestions[index]);
        return true;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setDismissed(true);
        return true;
      }
      return false;
    },
    [visible, suggestions, index, pick],
  );

  return { visible, suggestions, index, setIndex, pick, handleKeyDown };
}

export function FieldSuggestDropdown({
  suggestions,
  index,
  onPick,
  onHover,
  textareaRef,
}: {
  suggestions: Suggestion[];
  index: number;
  onPick: (s: Suggestion) => void;
  onHover: (i: number) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const measure = useCallback(() => {
    const el = textareaRef.current;
    if (el) setRect(el.getBoundingClientRect());
  }, [textareaRef]);

  useLayoutEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [measure]);

  // Keep the highlighted row in view.
  useEffect(() => {
    const row = ref.current?.querySelector<HTMLElement>(`[data-idx="${index}"]`);
    row?.scrollIntoView({ block: "nearest" });
  }, [index]);

  if (!rect || !suggestions.length) return null;

  // Anchor just above the composer, full composer width, viewport-relative so
  // the scroll container's overflow can't clip it.
  const width = Math.max(rect.width, 280);
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
  const bottom = window.innerHeight - rect.top + 8;

  return (
    <div
      ref={ref}
      className="fixed z-50 overflow-hidden rounded-xl border border-solid bg-background shadow-lg"
      style={{ width, left, bottom }}
      role="listbox"
      aria-label={T_FIELD}
    >
      <div className="border-b border-solid px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {T_FIELD}
      </div>
      <ul className="max-h-72 overflow-y-auto py-1">
        {suggestions.map((s, i) => (
          <li
            key={s.key}
            data-idx={i}
            role="option"
            aria-selected={i === index}
            onMouseEnter={() => onHover(i)}
            onMouseDown={(e) => {
              e.preventDefault(); // keep textarea focus
              onPick(s);
            }}
            className={cn(
              "flex cursor-pointer items-start gap-2 px-3 py-2",
              i === index ? "bg-accent" : "hover:bg-accent/60",
            )}
          >
            <span className="mt-0.5 shrink-0">
              {s.isTime ? (
                <Clock className="h-3.5 w-3.5 text-muted-foreground" />
              ) : s.isIdentifier ? (
                <Fingerprint className="h-3.5 w-3.5 text-muted-foreground" />
              ) : null}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-2">
                <span className="truncate text-sm font-medium">{s.label}</span>
                <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {s.field}
                </span>
              </span>
              {s.tableLabel ? (
                <span className="mt-0.5 block truncate text-[11px] text-muted-foreground/70">
                  {s.tableLabel}
                </span>
              ) : null}
              {s.thaiDescription ? (
                <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                  {s.thaiDescription}
                </span>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
