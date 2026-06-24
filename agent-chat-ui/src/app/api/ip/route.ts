import { NextResponse } from "next/server";

function extractClientIp(headers: Headers): string | null {
  const forwarded =
    headers.get("x-forwarded-for") ??
    headers.get("x-real-ip") ??
    headers.get("cf-connecting-ip") ??
    headers.get("x-client-ip") ??
    "";
  if (!forwarded) return null;
  const first = forwarded.split(",")[0]?.trim();
  return first || null;
}

export async function GET(request: Request) {
  const ip = extractClientIp(request.headers) ?? "unknown";
  return NextResponse.json({ ip }, { headers: { "Cache-Control": "no-store" } });
}
