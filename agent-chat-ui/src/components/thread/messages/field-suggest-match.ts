// Field-suggest matching: token extraction, tone-insensitive matching, and
// the matched-SPAN within the typed token.
//
// The span is what makes Thai insertion surgical: Thai has no spaces, so the
// typed token is a whole run ("แสดงคะแนนตามรายวิชา"), but only the part that
// actually matched a field ("คะแนน") is replaced when a suggestion is picked.
// English tokens are single words, so their span is the whole token.
//
// Matching is tone-mark-insensitive: the catalog's Thai descriptions are
// stored without tone marks (e.g. "หมายเลขบันทึก"), while users type with them
// ("หมายเลขบันทึก"). Normalizing both sides to base letters keeps the match
// working either way.

export type MatchSpan = { start: number; end: number };

export type SuggestColumnLike = {
  name: string;
  terms?: string[];
  thaiDescription?: string | null;
};

// Thai tone marks + phinthu + thanthakhat + sara-am (carry no identity of
// their own for matching): mai tho / mai ek / mai tri / mai chattawa /
// mai han-akat / maitaikhu / phinthu / sara am / thanthakhat.
const THAI_NONBASE_G = /[\u0e31\u0e34\u0e35\u0e36\u0e47\u0e48\u0e49\u0e4a\u0e4c]/g;
const THAI_NONBASE_ONE = /[\u0e31\u0e34\u0e35\u0e36\u0e47\u0e48\u0e49\u0e4a\u0e4c]/;

/** Lowercase, drop Thai tone/diacritic marks, collapse whitespace. */
export function normalizeForMatch(s: string): string {
  return s.toLowerCase().replace(THAI_NONBASE_G, "").replace(/\s+/g, " ").trim();
}

export function isThai(s: string): boolean {
  return /[\u0e00-\u0e7f]/.test(s);
}

/** The trailing run of word characters (ASCII alnum/underscore or Thai). */
export function trailingToken(text: string): { token: string; start: number } | null {
  const m = text.match(/([A-Za-z0-9\u0e00-\u0e7f_]+)$/);
  if (!m || m.index === undefined) return null;
  return { token: m[1], start: m.index };
}

/** True when the char is a "base" char that survives normalization. */
function isBaseChar(ch: string): boolean {
  return !THAI_NONBASE_ONE.test(ch) && /\S/.test(ch);
}

/** Map a span in normalized-token coordinates back to original coordinates. */
function normSpanToOriginal(token: string, nStart: number, nEnd: number): MatchSpan | null {
  let n = 0;
  let oStart: number | null = null;
  let oEnd: number | null = null;
  for (let i = 0; i < token.length; i++) {
    if (!isBaseChar(token[i])) continue;
    if (n === nStart && oStart === null) oStart = i;
    n++;
    if (n === nEnd) {
      oEnd = i + 1;
      break;
    }
  }
  if (oStart === null || oEnd === null) return null;
  return { start: oStart, end: oEnd };
}

/**
 * Best window of `token` matching one of the field's name/terms (tone-
 * insensitive). Returns the span in ORIGINAL token coordinates, or null.
 *
 * For each candidate phrase we look for the longest window of the token whose
 * normalized form equals the same-length prefix of the normalized phrase
 * (or, for a window ending at the token's end, is a prefix of the phrase).
 * The longest match wins; ties go to the earlier position.
 */
export function findMatchSpan(token: string, candidates: string[]): MatchSpan | null {
  if (!token) return null;
  const tNorm = normalizeForMatch(token);
  if (!tNorm) return null;
  let best: MatchSpan | null = null;
  for (const raw of candidates) {
    if (!raw) continue;
    const pNorm = normalizeForMatch(raw);
    if (!pNorm) continue;
    const maxLen = Math.min(tNorm.length, pNorm.length);
    for (let len = maxLen; len >= 2; len--) {
      let found: MatchSpan | null = null;
      for (let i = 0; i + len <= tNorm.length; i++) {
        const window = tNorm.slice(i, i + len);
        const hit =
          window === pNorm.slice(0, len) ||
          (i + len === tNorm.length && pNorm.startsWith(window));
        if (hit) {
          found = normSpanToOriginal(token, i, i + len);
          break; // earliest position at this length
        }
      }
      if (found) {
        if (!best || found.end - found.start > best.end - best.start) best = found;
        break; // only the longest window per candidate matters
      }
    }
  }
  return best;
}

/** Longest common substring of token vs a description, as a token span. */
function lcsSpan(token: string, desc: string): MatchSpan | null {
  const a = normalizeForMatch(token);
  const b = normalizeForMatch(desc);
  if (!a.length || !b.length) return null;
  let bestLen = 0;
  let bestI = 0;
  let prev = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1;
        if (cur[j] > bestLen) {
          bestLen = cur[j];
          bestI = i - cur[j];
        }
      }
    }
    prev = cur;
  }
  if (bestLen < 2) return null;
  return normSpanToOriginal(token, bestI, bestI + bestLen);
}

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

export function humanizeField(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
}

/**
 * Score a column against the typed token (tone-insensitive) and report the
 * span of the token that matched. The span is used for surgical replacement:
 * only the matched part of the typed run is swapped for the field label.
 */
export function scoreFieldWithSpan(
  col: SuggestColumnLike,
  token: string,
): { score: number; span: MatchSpan | null } {
  const t = normalizeForMatch(token);
  if (!t) return { score: 0, span: null };
  const name = normalizeForMatch(col.name);
  const human = normalizeForMatch(humanizeField(col.name));
  const terms = (col.terms ?? []).map((x) => normalizeForMatch(x));
  const thai = normalizeForMatch(col.thaiDescription ?? "");

  let score = 0;
  if (name.startsWith(t)) score = Math.max(score, 100);
  else if (name.includes(t)) score = Math.max(score, 70);
  if (human.startsWith(t)) score = Math.max(score, 95);
  else if (human.includes(t)) score = Math.max(score, 65);
  for (const term of terms) {
    if (!term) continue;
    if (term === t) score = Math.max(score, 90);
    else if (term.startsWith(t)) score = Math.max(score, 80);
    else if (term.includes(t)) score = Math.max(score, 60);
  }
  if (thai) {
    if (isThai(token)) {
      const shared = lcsLen(t, thai);
      if (shared >= 3) score = Math.max(score, 55 + shared * 5);
    } else if (thai.includes(t)) {
      score = Math.max(score, 78);
    }
  }
  if (score <= 0) return { score: 0, span: null };

  // The span to replace. English tokens are single words: replace the whole
  // token. Thai tokens are whole runs: replace only the matched part (a
  // name/term hit, else the longest-common window against the Thai
  // description, else the whole token as a last resort).
  let span: MatchSpan | null = { start: 0, end: token.length };
  if (isThai(token)) {
    span =
      findMatchSpan(token, [col.name, human, ...terms]) ??
      (thai ? lcsSpan(token, thai) : null) ??
      span;
  }
  return { score, span };
}
