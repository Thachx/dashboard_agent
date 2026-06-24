import type { Message } from "@langchain/langgraph-sdk";

const HIDDEN_UPLOAD_CONTEXT_REGEX =
  /\n*(?:Uploaded file context:[\s\S]*?(?:Use the uploaded file context above\. Do not search the filesystem for these uploaded files\.|$)|The uploaded file has already been processed by the application\.[\s\S]*?(?:When the user asks for text in the uploaded file, return only the text between FILE_TEXT markers\.|$))/gi;

export function stripHiddenUploadContext(text: string): string {
  return text.replace(HIDDEN_UPLOAD_CONTEXT_REGEX, "").trim();
}

/**
 * Extracts a string summary from a message's content, supporting multimodal (text, image, file, etc.).
 * - If text is present, returns the joined text.
 * - If not, returns a label for the first non-text modality (e.g., 'Image', 'Other').
 * - If unknown, returns 'Multimodal message'.
 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return stripHiddenUploadContext(content);
  if (!Array.isArray(content)) return "";
  const texts = content.flatMap((c) => {
    if (typeof c === "string") return [c];
    if (!c || typeof c !== "object") return [];
    if ("type" in c && c.type === "text" && typeof c.text === "string") {
      return [c.text];
    }
    if ("text" in c && typeof c.text === "string") {
      return [c.text];
    }
    if ("content" in c && typeof c.content === "string") {
      return [c.content];
    }
    if ("value" in c && typeof c.value === "string") {
      return [c.value];
    }
    return [];
  });
  return stripHiddenUploadContext(texts.join(" "));
}
