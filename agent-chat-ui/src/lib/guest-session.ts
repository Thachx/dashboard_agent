const THREAD_COOKIE_NAME = "lg_guest_thread_id";
const THREAD_OWNER_COOKIE_NAME = "lg_guest_thread_owner";
const THREAD_SET_AT_COOKIE_NAME = "lg_guest_thread_set_at";
const GUEST_COOKIE_NAME = "lg_guest_id";
const GUEST_SET_AT_COOKIE_NAME = "lg_guest_set_at";
const COOKIE_MAX_AGE_DAYS = 365;
const RENEW_THRESHOLD_DAYS = 30;

const hasDocument = typeof document !== "undefined";

function getCookie(name: string): string | null {
  if (!hasDocument) return null;
  const cookie = document.cookie
    .split(";")
    .map((value) => value.trim())
    .find((value) => value.startsWith(`${name}=`));
  if (!cookie) return null;
  const raw = cookie.slice(name.length + 1);
  return raw ? decodeURIComponent(raw) : null;
}

function setCookie(name: string, value: string, maxAgeSeconds: number) {
  if (!hasDocument) return;
  document.cookie = `${name}=${encodeURIComponent(
    value,
  )}; Max-Age=${maxAgeSeconds}; Path=/; SameSite=Lax`;
}

function getCookieNumber(name: string): number | null {
  const value = getCookie(name);
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function shouldRenew(setAtSeconds: number): boolean {
  const maxAgeSeconds = COOKIE_MAX_AGE_DAYS * 24 * 60 * 60;
  const renewThresholdSeconds = RENEW_THRESHOLD_DAYS * 24 * 60 * 60;
  return nowSeconds() - setAtSeconds >= maxAgeSeconds - renewThresholdSeconds;
}

function generateGuestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  const randomPart = Math.random().toString(16).slice(2);
  return `guest-${Date.now().toString(16)}-${randomPart}`;
}

export function getGuestThreadId(): string | null {
  const guestId = getCookie(GUEST_COOKIE_NAME);
  const ownerId = getCookie(THREAD_OWNER_COOKIE_NAME);
  if (guestId && ownerId && guestId !== ownerId) {
    return null;
  }
  if (!ownerId && guestId) {
    return null;
  }
  const threadId = getCookie(THREAD_COOKIE_NAME);
  if (threadId && guestId) {
    const setAt = getCookieNumber(THREAD_SET_AT_COOKIE_NAME);
    if (setAt == null || shouldRenew(setAt)) {
      setThreadCookies(threadId, guestId);
    }
  }
  return threadId;
}

export function setGuestThreadId(threadId: string, guestId?: string) {
  if (!hasDocument || !threadId) return;
  const resolvedGuestId = guestId || ensureGuestId();
  setThreadCookies(threadId, resolvedGuestId);
}

function setThreadCookies(threadId: string, guestId: string) {
  const maxAgeSeconds = COOKIE_MAX_AGE_DAYS * 24 * 60 * 60;
  setCookie(THREAD_COOKIE_NAME, threadId, maxAgeSeconds);
  setCookie(THREAD_OWNER_COOKIE_NAME, guestId, maxAgeSeconds);
  setCookie(THREAD_SET_AT_COOKIE_NAME, String(nowSeconds()), maxAgeSeconds);
}

export function clearGuestThreadId() {
  if (!hasDocument) return;
  document.cookie = `${THREAD_COOKIE_NAME}=; Max-Age=0; Path=/; SameSite=Lax`;
  document.cookie = `${THREAD_OWNER_COOKIE_NAME}=; Max-Age=0; Path=/; SameSite=Lax`;
  document.cookie = `${THREAD_SET_AT_COOKIE_NAME}=; Max-Age=0; Path=/; SameSite=Lax`;
}

export function getGuestId(): string | null {
  return getCookie(GUEST_COOKIE_NAME);
}

export function ensureGuestId(): string {
  if (!hasDocument) return "";
  const existing = getCookie(GUEST_COOKIE_NAME);
  if (existing) {
    const setAt = getCookieNumber(GUEST_SET_AT_COOKIE_NAME);
    if (setAt == null || shouldRenew(setAt)) {
      const maxAgeSeconds = COOKIE_MAX_AGE_DAYS * 24 * 60 * 60;
      setCookie(GUEST_COOKIE_NAME, existing, maxAgeSeconds);
      setCookie(GUEST_SET_AT_COOKIE_NAME, String(nowSeconds()), maxAgeSeconds);
    }
    return existing;
  }
  const next = generateGuestId();
  const maxAgeSeconds = COOKIE_MAX_AGE_DAYS * 24 * 60 * 60;
  setCookie(GUEST_COOKIE_NAME, next, maxAgeSeconds);
  setCookie(GUEST_SET_AT_COOKIE_NAME, String(nowSeconds()), maxAgeSeconds);
  return next;
}
