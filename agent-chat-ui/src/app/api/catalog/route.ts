import { NextResponse } from "next/server";

// Thin proxy to the local catalog API (field & table guide).
// The catalog service binds 127.0.0.1:8090 on the same host as Next.js,
// so this server-side route is the only network hop the browser sees.
const CATALOG_BASE = process.env.CATALOG_API_URL ?? "http://127.0.0.1:8090";

export const runtime = "nodejs";

export async function GET() {
  try {
    const upstream = await fetch(`${CATALOG_BASE}/api/catalog`, {
      cache: "no-store",
    });
    if (!upstream.ok) {
      return NextResponse.json(
        { error: `catalog upstream ${upstream.status}` },
        { status: upstream.status },
      );
    }
    const body = await upstream.json();
    return NextResponse.json(body);
  } catch (err) {
    return NextResponse.json(
      { error: "catalog unavailable" },
      { status: 502 },
    );
  }
}
