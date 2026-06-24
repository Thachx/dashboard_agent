import { NextRequest, NextResponse } from "next/server";
import { createRequire } from "node:module";

export const runtime = "nodejs";

const require = createRequire(import.meta.url);
const { PDFParse } = require("pdf-parse") as typeof import("pdf-parse");
const IMAGE_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

type ExtractFileRequest = {
  data?: unknown;
  mimeType?: unknown;
};

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as ExtractFileRequest;
    const data = typeof body.data === "string" ? body.data : "";
    const mimeType = typeof body.mimeType === "string" ? body.mimeType : "";

    if (!data) {
      return NextResponse.json({ error: "Missing file data." }, { status: 400 });
    }

    if (mimeType === "application/pdf") {
      const parser = new PDFParse({ data: Buffer.from(data, "base64") });
      const result = await parser.getText();
      await parser.destroy();

      return NextResponse.json({ text: result.text.trim() });
    }

    if (IMAGE_MIME_TYPES.has(mimeType)) {
      const { recognize } = await import("tesseract.js");
      const result = await recognize(Buffer.from(data, "base64"), "tha+eng");
      return NextResponse.json({ text: result.data.text.trim() });
    }

    if (mimeType !== "application/pdf") {
      return NextResponse.json(
        { error: `Unsupported file type: ${mimeType}` },
        { status: 400 },
      );
    }
  } catch (error) {
    console.error("Failed to extract file text:", error);
    return NextResponse.json(
      { error: "Failed to extract file text." },
      { status: 500 },
    );
  }
}
