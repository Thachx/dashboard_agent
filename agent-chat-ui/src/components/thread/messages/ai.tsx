import { parsePartialJson } from "@langchain/core/output_parsers";
import { useStreamContext } from "@/providers/Stream";
import {
  AIMessage,
  Checkpoint,
  Message,
  ToolMessage,
} from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { BranchSwitcher, CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";
import { cn } from "@/lib/utils";
import { ToolCalls, ToolResult } from "./tool-calls";
import { MessageContentComplex } from "@langchain/core/messages";
import { Fragment } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { useArtifact } from "../artifact";
import {
  SpreadsheetAnalysisVisual,
  type SpreadsheetAnalysisVisualPayload,
} from "./spreadsheet-analysis-visual";
import {
  PowerBiWidget,
  type GraphDashboardWidgetPayload,
} from "./power-bi-widget";

const SPREADSHEET_ANALYSIS_VISUAL_REGEX =
  /\n*<<<SPREADSHEET_ANALYSIS_VISUAL>>>\s*([\s\S]*?)\s*<<<END_SPREADSHEET_ANALYSIS_VISUAL>>>\s*/i;
const GRAPH_DASHBOARD_WIDGET_REGEX =
  /\n*<<<GRAPH_DASHBOARD_WIDGET>>>\s*([\s\S]*?)\s*<<<END_GRAPH_DASHBOARD_WIDGET>>>\s*/i;

function parseTimestamp(value: unknown): number | null {
  if (typeof value !== "string" || !value) return null;
  const ts = Date.parse(value);
  return Number.isNaN(ts) ? null : ts;
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 100) / 10);
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds - minutes * 60);
  const padded = seconds < 10 ? `0${seconds}` : `${seconds}`;
  return `${minutes}m ${padded}s`;
}

function CustomComponent({
  message,
  thread,
}: {
  message: Message;
  thread: ReturnType<typeof useStreamContext>;
}) {
  const artifact = useArtifact();
  const { values } = useStreamContext();
  const customComponents = values.ui?.filter(
    (ui) => ui.metadata?.message_id === message.id,
  );

  if (!customComponents?.length) return null;
  return (
    <Fragment key={message.id}>
      {customComponents.map((customComponent) => (
        <LoadExternalComponent
          key={customComponent.id}
          stream={thread as any}
          message={customComponent}
          meta={{ ui: customComponent, artifact }}
        />
      ))}
    </Fragment>
  );
}

function parseAnthropicStreamedToolCalls(
  content: MessageContentComplex[],
): AIMessage["tool_calls"] {
  const toolCallContents = content.filter((c) => c.type === "tool_use" && c.id);

  return toolCallContents.map((tc) => {
    const toolCall = tc as Record<string, any>;
    let json: Record<string, any> = {};
    if (toolCall?.input) {
      try {
        json = parsePartialJson(toolCall.input) ?? {};
      } catch {
        // Pass
      }
    }
    return {
      name: toolCall.name ?? "",
      id: toolCall.id ?? "",
      args: json,
      type: "tool_call",
    };
  });
}

function getToolCallsForMessage(
  msg: Message | undefined,
): AIMessage["tool_calls"] {
  if (!msg || msg.type !== "ai") return [];
  const msgContent = msg.content ?? [];
  const anthropicCalls = Array.isArray(msgContent)
    ? parseAnthropicStreamedToolCalls(msgContent as MessageContentComplex[])
    : undefined;
  const hasToolCalls =
    "tool_calls" in msg && msg.tool_calls && msg.tool_calls.length > 0;
  const toolCallsHaveContents =
    hasToolCalls &&
    msg.tool_calls?.some(
      (tc) => tc.args && Object.keys(tc.args).length > 0,
    );
  return (
    (hasToolCalls &&
      toolCallsHaveContents &&
      (msg.tool_calls as AIMessage["tool_calls"])) ||
    (anthropicCalls && anthropicCalls.length ? anthropicCalls : undefined) ||
    (hasToolCalls ? (msg.tool_calls as AIMessage["tool_calls"]) : undefined) ||
    []
  );
}

function separateThinking(
  text: string,
): { thinking: string; rest: string } {
  if (!text) return { thinking: "", rest: "" };
  const regex = /<thinking>([\s\S]*?)<\/thinking>/gi;
  const thinkingParts: string[] = [];
  const matches = Array.from(text.matchAll(regex));
  if (matches.length === 0 && /<thinking>/i.test(text)) {
    return {
      thinking: "",
      rest: text.replace(/<\/?thinking>/gi, "").trim(),
    };
  }
  const restWithoutClosed = text.replace(regex, (_match, inner) => {
    const trimmed = (inner || "").trim();
    if (trimmed) {
      thinkingParts.push(trimmed);
    }
    return "";
  });
  const openTag = "<thinking>";
  const lastOpenIdx = restWithoutClosed.lastIndexOf(openTag);
  const [rest, trailingThinking] =
    lastOpenIdx !== -1
      ? [
          restWithoutClosed.slice(0, lastOpenIdx),
          restWithoutClosed.slice(lastOpenIdx + openTag.length),
        ]
      : [restWithoutClosed, ""];
  const trailingTrimmed = trailingThinking.trim();
  if (trailingTrimmed) {
    thinkingParts.push(trailingTrimmed);
  }
  return {
    thinking: thinkingParts.join("\n\n").trim(),
    rest: rest.trim(),
  };
}

function injectSourceLinks(text: string): string {
  if (!text) return "";
  const sourceRegex = /\[(?:WEB\s+)?SOURCE:[^\]]+\]/gi;
  const matches = Array.from(text.matchAll(sourceRegex));
  if (!matches.length) return text;
  let cursor = 0;
  let result = "";
  let i = 0;
  while (i < matches.length) {
    const match = matches[i];
    const start = match.index ?? 0;
    const sources: string[] = [match[0]];
    let end = start + match[0].length;
    i += 1;
    while (i < matches.length) {
      const next = matches[i];
      const gap = text.slice(end, next.index ?? end);
      if (!/^\s*$/.test(gap)) break;
      sources.push(next[0]);
      end = (next.index ?? end) + next[0].length;
      i += 1;
    }
    result += text.slice(cursor, start);
    const encoded = encodeURIComponent(JSON.stringify(sources));
    result += `[source](source:${encoded})`;
    cursor = end;
  }
  result += text.slice(cursor);
  return result;
}

function extractSpreadsheetAnalysisVisual(
  text: string,
): { visibleText: string; visual: SpreadsheetAnalysisVisualPayload | null } {
  if (!text) return { visibleText: "", visual: null };
  const match = text.match(SPREADSHEET_ANALYSIS_VISUAL_REGEX);
  if (!match) {
    return { visibleText: text, visual: null };
  }
  const visibleText = text.replace(SPREADSHEET_ANALYSIS_VISUAL_REGEX, "\n\n").trim();
  try {
    const parsed = JSON.parse(match[1]) as SpreadsheetAnalysisVisualPayload;
    if (parsed?.kind === "spreadsheet-analysis-visual" && Array.isArray(parsed?.charts)) {
      return { visibleText, visual: parsed };
    }
  } catch {
    // ignore malformed hidden payloads and fall back to plain text
  }
  return { visibleText, visual: null };
}

function extractGraphDashboardWidget(
  text: string,
): { visibleText: string; widget: GraphDashboardWidgetPayload | null } {
  if (!text) return { visibleText: "", widget: null };
  const match = text.match(GRAPH_DASHBOARD_WIDGET_REGEX);
  if (!match) return { visibleText: text, widget: null };
  const visibleText = text.replace(GRAPH_DASHBOARD_WIDGET_REGEX, "\n\n").trim();
  try {
    const parsed = JSON.parse(match[1]) as GraphDashboardWidgetPayload;
    if (parsed?.kind === "graph-dashboard-widget") {
      return { visibleText, widget: parsed };
    }
  } catch {
    // Ignore malformed hidden payloads and keep the text response.
  }
  return { visibleText, widget: null };
}

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
  threadMessages: threadMessagesOverride,
}: {
  message: Message | undefined;
  isLoading: boolean;
  handleRegenerate: (parentCheckpoint: Checkpoint | null | undefined) => void;
  threadMessages?: Message[];
}) {
  const content = message?.content ?? [];
  const rawContentString = getContentString(content);
  const { thinking, rest } = separateThinking(rawContentString);
  const { visibleText: withoutDashboard, widget: graphDashboardWidget } =
    useMemo(() => extractGraphDashboardWidget(rest), [rest]);
  const { visibleText: contentString, visual: spreadsheetVisual } =
    useMemo(
      () => extractSpreadsheetAnalysisVisual(withoutDashboard),
      [withoutDashboard],
    );
  const [hideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const isThinkingMessage =
    (message as { type?: string } | undefined)?.type === "ai_thinking";

  const thread = useStreamContext();
  const threadMessages = useMemo(
    () => threadMessagesOverride ?? thread.messages ?? [],
    [threadMessagesOverride, thread.messages],
  );
  const isLastMessage =
    threadMessages[threadMessages.length - 1]?.id === message?.id;
  const hasNoAIOrToolMessages = !threadMessages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );
  const meta = message ? thread.getMessagesMetadata(message) : undefined;

  const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;
  const anthropicStreamedToolCalls = Array.isArray(content)
    ? parseAnthropicStreamedToolCalls(content)
    : undefined;

  const hasToolCalls =
    message &&
    "tool_calls" in message &&
    message.tool_calls &&
    message.tool_calls.length > 0;
  const toolCallsHaveContents =
    hasToolCalls &&
    message.tool_calls?.some(
      (tc) => tc.args && Object.keys(tc.args).length > 0,
    );
  const hasAnthropicToolCalls = !!anthropicStreamedToolCalls?.length;
  const isToolResult = message?.type === "tool";
  const isToolCallMessage =
    message?.type === "ai" && (hasToolCalls || hasAnthropicToolCalls);
  const messageIndex = useMemo(() => {
    if (!message?.id) return -1;
    return threadMessages.findIndex((m) => m.id === message.id);
  }, [threadMessages, message?.id]);

  const nextHumanIndex = useMemo(() => {
    if (messageIndex === -1) return -1;
    for (let i = messageIndex + 1; i < threadMessages.length; i++) {
      if (threadMessages[i].type === "human") {
        return i;
      }
    }
    return -1;
  }, [messageIndex, threadMessages]);

  const hasLaterAiBeforeNextHuman = useMemo(() => {
    if (messageIndex === -1) return false;
    const end =
      nextHumanIndex === -1 ? threadMessages.length : nextHumanIndex;
    for (let i = messageIndex + 1; i < end; i++) {
      if (
        threadMessages[i].type === "ai" ||
        threadMessages[i].type === "tool"
      ) {
        return true;
      }
    }
    return false;
  }, [messageIndex, nextHumanIndex, threadMessages]);
  const isFinalAiMessage = !hasLaterAiBeforeNextHuman;

  const previousHumanIndex = useMemo(() => {
    if (messageIndex <= 0) return -1;
    for (let i = messageIndex - 1; i >= 0; i--) {
      if (threadMessages[i].type === "human") {
        return i;
      }
    }
    return -1;
  }, [messageIndex, threadMessages]);

  const turnStartIndex = Math.max(previousHumanIndex + 1, 0);
  const turnEndIndex =
    nextHumanIndex === -1 ? threadMessages.length : nextHumanIndex;

  const detailEntries = useMemo(() => {
    const entries: Array<
      | { type: "reasoning"; content: string; key: string }
      | { type: "tool_use"; toolCalls: AIMessage["tool_calls"]; key: string }
      | { type: "tool_result"; toolMessage: ToolMessage; key: string }
    > = [];

    const pushReasoning = (content: string, key: string) => {
      if (content) {
        entries.push({ type: "reasoning", content, key });
      }
    };
    const pushToolCalls = (
      toolCalls: AIMessage["tool_calls"],
      key: string,
    ) => {
      if (toolCalls && toolCalls.length) {
        entries.push({ type: "tool_use", toolCalls, key });
      }
    };
    const pushToolResult = (toolMessage: ToolMessage, key: string) => {
      entries.push({ type: "tool_result", toolMessage, key });
    };

    const processAiMessage = (msg: Message, idx: number) => {
      const contentStr = getContentString(msg.content ?? []);
      if ((msg as { type?: string }).type === "ai_thinking") {
        pushReasoning(contentStr.trim(), `reasoning-${msg.id ?? idx}`);
        return;
      }
      const { thinking: msgThinking } = separateThinking(contentStr);
      pushReasoning(msgThinking, `reasoning-${msg.id ?? idx}`);
      if (!hideToolCalls) {
        const calls = getToolCallsForMessage(msg);
        pushToolCalls(calls, `toolcalls-${msg.id ?? idx}`);
      }
    };

    if (!message || message.type !== "ai" || messageIndex === -1) {
      return entries;
    }
    if (!isFinalAiMessage) {
      return entries;
    }

    for (let i = turnStartIndex; i < turnEndIndex; i++) {
      const msg = threadMessages[i];
      if (msg.type === "ai" || (msg as { type?: string }).type === "ai_thinking") {
        processAiMessage(msg, i);
      } else if (msg.type === "tool" && !hideToolCalls) {
        pushToolResult(msg as ToolMessage, `tool-result-${msg.id ?? i}`);
      }
    }

    return entries;
  }, [
    hideToolCalls,
    isFinalAiMessage,
    message,
    messageIndex,
    threadMessages,
    turnEndIndex,
    turnStartIndex,
  ]);

  const hasDetailsPanel =
    isFinalAiMessage && detailEntries.length > 0 && !isLoading;
  const thoughtDurationLabel = useMemo(() => {
    if (!message || message.type !== "ai" || messageIndex === -1) {
      return null;
    }
    if (!isFinalAiMessage) {
      return null;
    }
    const endMeta = thread.getMessagesMetadata(message, messageIndex);
    const endTs = parseTimestamp(endMeta?.firstSeenState?.created_at);
    if (!endTs) {
      return null;
    }
    let startTs: number | null = null;
    if (previousHumanIndex !== -1) {
      const humanMsg = threadMessages[previousHumanIndex];
      const humanMeta = thread.getMessagesMetadata(
        humanMsg,
        previousHumanIndex,
      );
      startTs = parseTimestamp(humanMeta?.firstSeenState?.created_at);
    }
    if (!startTs) {
      for (let i = turnStartIndex; i <= messageIndex; i++) {
        const msg = threadMessages[i];
        if (msg.type === "ai" || (msg as { type?: string }).type === "ai_thinking") {
          const meta = thread.getMessagesMetadata(msg, i);
          const ts = parseTimestamp(meta?.firstSeenState?.created_at);
          if (ts) {
            startTs = ts;
            break;
          }
        }
      }
    }
    if (!startTs) {
      return null;
    }
    return formatElapsed(endTs - startTs);
  }, [
    isFinalAiMessage,
    message,
    messageIndex,
    previousHumanIndex,
    thread,
    threadMessages,
    turnStartIndex,
  ]);
  const contentWithSources = useMemo(
    () => injectSourceLinks(contentString),
    [contentString],
  );

  useEffect(() => {
    setShowDetailPanel(false);
  }, [message?.id]);

  const aiMissingFromThread = message?.type === "ai" && messageIndex === -1;
  const isIntermediateAi = message?.type === "ai" && !isFinalAiMessage;
  const shouldHideStreamingAi =
    isLoading &&
    message?.type === "ai" &&
    nextHumanIndex === -1 &&
    isFinalAiMessage;
  const hasRenderableContent = contentString.trim().length > 0;
  const isThinkingOnlyContent =
    thinking.trim().length > 0 && !hasRenderableContent;
  if (
    isToolResult ||
    aiMissingFromThread ||
    isIntermediateAi ||
    shouldHideStreamingAi ||
    isToolCallMessage ||
    isThinkingMessage ||
    isThinkingOnlyContent
  ) {
    return null;
  }

  return (
    <div className="group mr-auto flex w-full items-start gap-2">
      <div className="flex w-full flex-col gap-2">
        {isToolResult ? (
          <>
            <ToolResult message={message as ToolMessage} />
          </>
        ) : (
          <>
            {hasDetailsPanel && !isLoading && (
              <div className="py-1 space-y-1">
                <button
                  type="button"
                  onClick={() => setShowDetailPanel((prev) => !prev)}
                  className="inline-flex items-center whitespace-nowrap text-[11px] uppercase tracking-tight text-muted-foreground/70 transition-colors hover:text-muted-foreground underline decoration-dotted"
                  style={{ whiteSpace: "nowrap", wordBreak: "keep-all" }}
                >
                  {thoughtDurationLabel
                    ? `Thought for ${thoughtDurationLabel}`
                    : "Thought for --"}
                </button>
                {showDetailPanel && (
                  <div className="mt-1 space-y-3 rounded-md border bg-muted/30 p-3 text-sm text-muted-foreground/80 max-h-96 overflow-y-auto w-full min-w-0">
                    {detailEntries.map((entry) => {
                      if (entry.type === "reasoning") {
                        return (
                          <div key={entry.key}>
                            <p className="text-xs font-semibold uppercase">
                              Reasoning
                            </p>
                            <div className="mt-1">
                              <MarkdownText>{entry.content}</MarkdownText>
                            </div>
                          </div>
                        );
                      }
                      if (entry.type === "tool_use") {
                        return (
                          <div key={entry.key}>
                            <p className="text-xs font-semibold uppercase">
                              Tool use
                            </p>
                            <div className="mt-1">
                              <ToolCalls toolCalls={entry.toolCalls} />
                            </div>
                          </div>
                        );
                      }
                      return (
                        <div key={entry.key}>
                          <p className="text-xs font-semibold uppercase">
                            Tool result
                          </p>
                          <div className="mt-1 space-y-2">
                            <ToolResult message={entry.toolMessage} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            {hasRenderableContent && (
              <div className="py-1">
                <MarkdownText>{contentWithSources}</MarkdownText>
              </div>
            )}
            {spreadsheetVisual && (
              <SpreadsheetAnalysisVisual payload={spreadsheetVisual} />
            )}
            {graphDashboardWidget && (
              <PowerBiWidget payload={graphDashboardWidget} />
            )}

            {!hideToolCalls && !hasDetailsPanel && !isLoading && (
              <>
                {(hasToolCalls && toolCallsHaveContents && (
                  <ToolCalls toolCalls={message.tool_calls} />
                )) ||
                  (hasAnthropicToolCalls && (
                    <ToolCalls toolCalls={anthropicStreamedToolCalls} />
                  )) ||
                  (hasToolCalls && (
                    <ToolCalls toolCalls={message.tool_calls} />
                  ))}
              </>
            )}

            {message && (
              <CustomComponent
                message={message}
                thread={thread}
              />
            )}
            <div
              className={cn(
                "mr-auto flex items-center gap-2 transition-opacity",
                "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
              )}
            >
              <BranchSwitcher
                branch={meta?.branch}
                branchOptions={meta?.branchOptions}
                onSelect={(branch) => thread.setBranch(branch)}
                isLoading={isLoading}
              />
              <CommandBar
                content={contentString}
                isLoading={isLoading}
                isAiMessage={true}
                handleRegenerate={() => handleRegenerate(parentCheckpoint)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function AssistantMessageLoading() {
  return (
    <div className="mr-auto flex items-start gap-2">
      <div className="bg-muted flex h-8 items-center gap-1 rounded-2xl px-4 py-2">
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full"></div>
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_0.5s_infinite] rounded-full"></div>
        <div className="bg-foreground/50 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_1s_infinite] rounded-full"></div>
      </div>
    </div>
  );
}

export function AssistantMessageError({
  error,
}: {
  error: string;
}) {
  return (
    <div className="mr-auto flex w-full items-start gap-2">
      <div className="flex w-full flex-col gap-2">
        <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
          <div className="font-medium">The run ended before the assistant could return a final response.</div>
          <div className="mt-1 whitespace-pre-wrap break-words text-rose-800/90">
            {error}
          </div>
        </div>
      </div>
    </div>
  );
}
