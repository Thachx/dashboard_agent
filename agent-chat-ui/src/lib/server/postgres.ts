import { Pool } from "pg";

function sanitizeConnectionString(conn?: string | null) {
  if (!conn) return conn ?? undefined;
  return conn
    .replace("postgresql+psycopg", "postgresql")
    .replace("postgresql+asyncpg", "postgresql")
    .replace("postgresql+pg8000", "postgresql");
}

export const PG_CONNECTION_STRING = sanitizeConnectionString(process.env.PG_CONN_STR);

const rawTableName = process.env.PG_CHAT_HISTORY_TABLE ?? "chat_message_history";
export const CHAT_HISTORY_TABLE =
  /^[A-Za-z0-9_]+$/.test(rawTableName) ? rawTableName : "chat_message_history";

let pool: Pool | null = null;

export function getPgPool(): Pool {
  if (!PG_CONNECTION_STRING) {
    throw new Error("PG_CONN_STR not configured");
  }
  if (!pool) {
    pool = new Pool({ connectionString: PG_CONNECTION_STRING });
  }
  return pool;
}

export function isPgConfigured(): boolean {
  return Boolean(PG_CONNECTION_STRING);
}
