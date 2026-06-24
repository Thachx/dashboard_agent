import { Client } from "@langchain/langgraph-sdk";

export function resolveApiUrl(apiUrl: string): string {
  if (/^https?:\/\//i.test(apiUrl)) {
    return apiUrl;
  }

  if (typeof window !== "undefined") {
    return new URL(apiUrl, window.location.origin).toString();
  }

  return apiUrl;
}

export function createClient(apiUrl: string, apiKey: string | undefined) {
  return new Client({
    apiKey,
    apiUrl: resolveApiUrl(apiUrl),
  });
}
