import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef, useCallback, useMemo, useState, FormEvent } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { Button } from "../ui/button";
import type {
  Message,
  Thread as LangGraphThread,
} from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageError, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { DO_NOT_RENDER_ID_PREFIX } from "@/lib/ensure-tool-responses";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";
import { clearGuestThreadId, ensureGuestId } from "@/lib/guest-session";
import { useThreads } from "@/providers/Thread";
import { getContentString } from "./utils";
import { ThreadView } from "./agent-inbox";
import { isAgentInboxInterruptSchema } from "@/lib/agent-inbox-interrupt";
import { GenericInterruptView } from "./messages/generic-interrupt";
import { MarkdownText } from "./markdown-text";

const isThreadNotFoundError = (message: string) => {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("thread not found") ||
    (normalized.includes("not found") && normalized.includes("thread"))
  );
};

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [userId, setUserId] = useState("");
  const userIdRef = useRef("");
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const { setThreads } = useThreads();

  const stream = useStreamContext();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messages = useMemo(() => {
    const raw = stream.messages ?? [];
    const lastIndexById = new Map<string, number>();
    raw.forEach((msg, index) => {
      const id = String(msg?.id ?? "").trim();
      if (id) {
        lastIndexById.set(id, index);
      }
    });

    const deduped: Message[] = [];
    for (let index = 0; index < raw.length; index += 1) {
      const msg = raw[index];
      const id = String(msg?.id ?? "").trim();
      if (id && lastIndexById.get(id) !== index) {
        continue;
      }
      deduped.push(msg);
    }
    return deduped;
  }, [stream.messages]);
  const isLoading = stream.isLoading;
  const stableMessagesRef = useRef<Message[]>([]);
  useEffect(() => {
    if (!isLoading) {
      stableMessagesRef.current = messages;
    }
  }, [isLoading, messages]);
  const displayMessages = useMemo(() => {
    if (!isLoading) {
      return messages;
    }

    const stableMessages = stableMessagesRef.current;
    if (!stableMessages.length) {
      return messages;
    }

    const currentById = new Map(
      messages
        .filter((message) => message.id)
        .map((message) => [String(message.id), message] as const),
    );
    const stableIds = new Set<string>();
    const merged = stableMessages.map((message) => {
      const id = message.id ? String(message.id) : "";
      if (!id) {
        return message;
      }
      stableIds.add(id);
      return currentById.get(id) ?? message;
    });

    for (const message of messages) {
      const id = message.id ? String(message.id) : "";
      if (!id || !stableIds.has(id)) {
        merged.push(message);
      }
    }

    return merged;
  }, [isLoading, messages]);
  const stripThinking = (text: string) =>
    text.replace(/<thinking>[\s\S]*?<\/thinking>/gi, "").trim();
  const lastHumanIndex = (() => {
    for (let i = displayMessages.length - 1; i >= 0; i--) {
      if (displayMessages[i].type === "human") return i;
    }
    return -1;
  })();
  const hasVisibleAiAfterHuman = displayMessages
    .slice(lastHumanIndex + 1)
    .some((m) => {
      if (m.type !== "ai") return false;
      if (isLoading) return false;
      if ("tool_calls" in m && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
        return false;
      }
      const content = getContentString(m.content ?? []);
      return stripThinking(content).length > 0;
    });
  const showLoadingBubble = isLoading && !hasVisibleAiAfterHuman;
  const threadErrorMessage = !isLoading && stream.error
    ? String((stream.error as { message?: unknown })?.message ?? "An unknown error occurred.")
    : "";
  const showInlineError =
    threadErrorMessage.length > 0 && !hasVisibleAiAfterHuman && !stream.interrupt;

  const lastError = useRef<string | undefined>(undefined);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "0px";
    const nextHeight = Math.min(textarea.scrollHeight, 288);
    textarea.style.height = `${Math.max(nextHeight, 96)}px`;
    textarea.style.overflowY = textarea.scrollHeight > 288 ? "auto" : "hidden";
  }, [input]);

  const resolveUserId = useCallback(async () => {
    if (userIdRef.current) return userIdRef.current;
    try {
      const res = await fetch("/api/ip", { cache: "no-store" });
      if (res.ok) {
        const data = (await res.json()) as { ip?: unknown };
        const ip = typeof data?.ip === "string" ? data.ip.trim() : "";
        if (ip) {
          userIdRef.current = ip;
          setUserId(ip);
          return ip;
        }
      }
    } catch (error) {
      console.error("Failed to resolve user IP:", error);
    }
    const fallback = "unknown";
    userIdRef.current = fallback;
    setUserId(fallback);
    return fallback;
  }, []);
  const parseApprovalDecision = (text: string) => {
    const normalized = text.trim().toLowerCase();
    if (!normalized) return null;
    if (
      ["approve", "approved", "ok", "okay", "yes", "y", "allow"].includes(
        normalized,
      )
    ) {
      return { type: "approve" } as const;
    }
    if (["deny", "denied", "no", "n", "reject"].includes(normalized)) {
      return { type: "reject", message: "User denied web search." } as const;
    }
    return null;
  };

  const setThreadId = useCallback((id: string | null) => {
    _setThreadId(id);
    if (!id) {
      clearGuestThreadId();
    }

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  }, [_setThreadId, closeArtifact, setArtifactContext]);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      if (isThreadNotFoundError(String(message))) {
        lastError.current = message;
        setThreadId(null);
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [setThreadId, stream.error]);

  useEffect(() => {
    void resolveUserId();
  }, [resolveUserId]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (input.trim().length === 0 || isLoading) return;

    const trimmedInput = input.trim();

    const resolvedUserId = await resolveUserId();
    const interrupt = stream.interrupt;
    const approvalDecision =
      interrupt && isAgentInboxInterruptSchema(interrupt)
        ? parseApprovalDecision(input)
        : null;
    const firstInterrupt = Array.isArray(interrupt) ? interrupt[0] : interrupt;
    const firstInterruptValue = firstInterrupt?.value as
      | { action_requests?: { name?: string }[] }
      | undefined;
    const isWebApproval =
      approvalDecision &&
      firstInterruptValue?.action_requests?.[0]?.name === "web_search";
    if (isWebApproval) {
      stream.submit(
        {},
        {
          config: {
            configurable: { web_search_enabled: false },
          },
          metadata: {
            user_id: resolvedUserId,
            web_search_enabled: false,
          },
          command: {
            resume: { decisions: [approvalDecision] },
          },
        },
      );
      setInput("");
      return;
    }

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [{ type: "text", text: trimmedInput }] as Message["content"],
    };

    const context = {
      ...(Object.keys(artifactContext).length > 0 ? artifactContext : {}),
      web_search_enabled: false,
    };

    const guestId = ensureGuestId();
    const needsThreadId = !threadId;
    const newThreadId = needsThreadId ? uuidv4() : undefined;
    const threadName =
      getContentString(newHumanMessage.content).trim() || "New thread";
    const threadMetadata = needsThreadId
      ? { guest_id: guestId, thread_name: threadName }
      : { guest_id: guestId };
    const submitMetadata = {
      ...threadMetadata,
      web_search_enabled: false,
      user_id: resolvedUserId,
    };

    if (needsThreadId && newThreadId) {
      const nowIso = new Date().toISOString();
      const placeholder: LangGraphThread = {
        thread_id: newThreadId,
        created_at: nowIso,
        updated_at: nowIso,
        state_updated_at: nowIso,
        metadata: threadMetadata,
        status: "busy",
        values: { messages: [newHumanMessage] },
        interrupts: {},
      };
      setThreads((prev) =>
        prev.some((t) => t.thread_id === newThreadId)
          ? prev
          : [placeholder, ...prev],
      );
    }

    stream.submit(
      { messages: [newHumanMessage], context },
      {
        streamMode: ["values"],
        streamSubgraphs: false,
        streamResumable: false,
        config: {
          configurable: {
            web_search_enabled: false,
          },
        },
        metadata: submitMetadata,
        threadId: newThreadId,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
  };

  const handleRegenerate = (humanPrompt: string) => {
    const trimmedPrompt = humanPrompt.trim();
    if (!trimmedPrompt) {
      toast.error("Could not find the prompt to refresh.", {
        duration: 3000,
      });
      return;
    }
    const resolvedUserId = userIdRef.current || userId || "unknown";
    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [{ type: "text", text: trimmedPrompt }] as Message["content"],
    };
    const context = {
      web_search_enabled: false,
    };
    stream.submit(
      { messages: [newHumanMessage], context },
      {
        streamMode: ["values"],
        streamSubgraphs: false,
        streamResumable: false,
        config: {
          configurable: {
            web_search_enabled: false,
          },
        },
        metadata: {
          guest_id: ensureGuestId(),
          web_search_enabled: false,
          user_id: resolvedUserId,
        },
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            newHumanMessage,
          ],
        }),
      },
    );
  };

  const chatStarted = !!threadId || !!displayMessages.length;
  const approvalPromptText = () => {
    const interrupt = stream.interrupt;
    if (!interrupt || !isAgentInboxInterruptSchema(interrupt)) return "";
    const first = Array.isArray(interrupt) ? interrupt[0] : interrupt;
    const description =
      first?.value?.action_requests?.[0]?.description?.trim() ?? "";
    const cleaned = stripThinking(description);
    if (cleaned) return cleaned;
    const lastAiMessage = [...messages]
      .reverse()
      .find((m) => m.type === "ai");
    const aiContent = lastAiMessage
      ? stripThinking(getContentString(lastAiMessage.content ?? []))
      : "";
    return aiContent || "Please approve the web search.";
  };

  const shouldRenderApprovalPrompt = () => {
    const interrupt = stream.interrupt;
    if (!interrupt || !isAgentInboxInterruptSchema(interrupt)) return false;
    if (!messages.length) return true;
    const lastAiMessage = [...messages]
      .reverse()
      .find((m) => m.type === "ai");
    const lastAiText = lastAiMessage
      ? stripThinking(getContentString(lastAiMessage.content ?? []))
      : "";
    const prompt = approvalPromptText();
    return !lastAiText || lastAiText !== prompt;
  };

  const renderApprovalPrompt = () => {
    const prompt = approvalPromptText();
    if (!prompt) return null;
    return (
      <div className="mr-auto flex w-full items-start gap-2">
        <div className="flex w-full flex-col gap-2">
          <div className="py-1">
            <MarkdownText>{prompt}</MarkdownText>
          </div>
        </div>
      </div>
    );
  };

  const renderInterrupt = () => {
    const interrupt = stream.interrupt;
    if (!interrupt) return null;
    const fallbackValue = Array.isArray(interrupt)
      ? (interrupt as Record<string, any>[])
      : (((interrupt as { value?: unknown } | undefined)?.value ??
          interrupt) as Record<string, any>);
    return isAgentInboxInterruptSchema(interrupt) ? (
      <ThreadView interrupt={interrupt} />
    ) : (
      <GenericInterruptView interrupt={fallbackValue} />
    );
  };

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="relative h-full"
            style={{ width: 300 }}
          >
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center gap-3 px-4 py-2">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                  ) : (
                    <PanelRightClose className="size-5" />
                  )}
                </Button>
                  )}
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2 pr-16">
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
                    <Button
                      className="hover:bg-gray-100"
                      variant="ghost"
                      onClick={() => setChatHistoryOpen((p) => !p)}
                    >
                      {chatHistoryOpen ? (
                        <PanelRightOpen className="size-5" />
                      ) : (
                        <PanelRightClose className="size-5" />
                      )}
                    </Button>
                  )}
                </div>
                <motion.button
                  className="flex cursor-pointer items-center gap-2"
                  onClick={() => setThreadId(null)}
                  animate={{
                    marginLeft: !chatHistoryOpen ? 48 : 0,
                  }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 30,
                  }}
                >
                  <span className="text-xl font-semibold tracking-tight">
                    Dashboard Graph Agent
                  </span>
                </motion.button>
              </div>

              <div className="flex items-center gap-2">
                <TooltipIconButton
                  className="size-9 p-0"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 mx-auto flex w-full max-w-[1600px] flex-col gap-4"
              content={
                <>
                  {displayMessages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) =>
                      message.type === "human" ? (
                        <HumanMessage
                          key={message.id ? `${message.id}-${index}` : `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                        />
                    ) : (
                        <AssistantMessage
                          key={message.id ? `${message.id}-${index}` : `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                          threadMessages={displayMessages}
                          handleRegenerate={handleRegenerate}
                        />
                      ),
                    )}
                  {!!stream.interrupt && (
                    <>
                      {shouldRenderApprovalPrompt() && renderApprovalPrompt()}
                      <div className="w-full">{renderInterrupt()}</div>
                    </>
                  )}
                  {showInlineError && (
                    <AssistantMessageError error={threadErrorMessage} />
                  )}
                  {showLoadingBubble && <AssistantMessageLoading />}
                </>
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-white">
                  {!chatStarted && (
                    <div className="flex items-center gap-3">
                      <h1 className="text-2xl font-semibold tracking-tight">
                        Agent Chat
                      </h1>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div className="bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl border border-solid shadow-xs">
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <textarea
                        ref={textareaRef}
                        rows={4}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="composer-scrollbar min-h-24 max-h-72 w-full resize-none overflow-hidden border-none bg-transparent p-3.5 pb-0 leading-6 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div className="flex items-center gap-3 p-2 pt-4">
                        {stream.isLoading ? (
                          <Button
                            key="stop"
                            onClick={() => stream.stop()}
                            className="ml-auto"
                          >
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="ml-auto shadow-md transition-all"
                            disabled={
                              isLoading ||
                              !input.trim()
                            }
                          >
                            Send
                          </Button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
    </div>
  );
}
