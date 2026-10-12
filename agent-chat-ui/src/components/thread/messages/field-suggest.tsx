"use client";
// Field autosuggest for the chat composer.
//
// General, data-driven: it fetches the field catalog once, then matches the
// user's ACTIVE TYPED TOKEN against every field's name, English terms, and Thai
// description. It works for English ("grade", "user") and Thai ("คะแนน") alike
// because the token is just the trailing run of word characters (ASCII or Thai
// script) — no per-field or per-language hardcoding. Picking a field inserts its
// readable name so the question names the field explicitly.
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
};

// House spelling (same codepoints as the data guide) for the panel header.
const T_FIELD = "\u0e1f\u0e34\u0e25\u0e14\u0e4c"; // field

function humanizeField(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
}

// The trailing run of word characters (ASCII alnum/underscore or Thai block).
// English: last space-delimited word. Thai: the whole typed run (no spaces).
function trailingToken(text: string): { token: string; start: number } | null {
  const m = text.match(/([A-Za-z0-9\u0e00-\u0e7f_]+)$/);
  if (!m || m.index === undefined) return null;
  return { token: m[1], start: m.index };
}

// Thai has no word boundaries, so a typed Thai token is a whole run and a plain
// substring test misses ("แสดงคะแนน" does not contain the description, but
// shares "คะแนน"). Longest-common-substring length is a language-neutral proxy
// for how much of the typed Thai actually appears in the field's Thai label.
function lcsLen(a: string, b: string): number {
  if (!a.length || !b.length) return 0;
  let prev = new Array(b.length + 1).fill(0);
  let best = 0;
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1;
        if (cur[j] > best) best = cur[j];
      }
    }
    prev = cur;
  }
  return best;
}

function isThai(s: string): boolean {
  return /[\u0e00-\u0e7f]/.test(s);
}

function scoreField(col: SuggestColumn, token: string): number {
  const t = token.toLowerCase();
  const name = col.name.toLowerCase();
  const human = humanizeField(col.name).toLowerCase();
  const terms = (col.terms ?? []).map((x) => x.toLowerCase());
  const thai = col.thaiDescription ?? "";
  let score = 0;
  if (name.startsWith(t)) score = Math.max(score, 100);
  else if (name.includes(t)) score = Math.max(score, 70);
  if (human.startsWith(t)) score = Math.max(score, 95);
  else if (human.includes(t)) score = Math.max(score, 65);
  for (const term of terms) {
    if (term === t) score = Math.max(score, 90);
    else if (term.startsWith(t)) score = Math.max(score, 80);
    else if (term.includes(t)) score = Math.max(score, 60);
  }
  if (thai) {
    if (isThai(token)) {
      const shared = lcsLen(token, thai);
      if (shared >= 3) score = Math.max(score, 55 + shared * 5);
    } else if (thai.includes(token)) {
      score = Math.max(score, 78);
    }
  }
  return score;
}

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
      const score = scoreField(col, token);
      if (score <= 0) continue;
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
      const next =
        input.slice(0, info.start) +
        s.label +
        " " +
        input.slice(info.start + info.token.length);
      setInput(next);
      setDismissed(true);
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        const pos = info.start + s.label.length + 1;
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
