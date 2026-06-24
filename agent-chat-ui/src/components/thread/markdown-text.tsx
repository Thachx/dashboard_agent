"use client";

import "./markdown-styles.css";

import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { FC, memo, useState } from "react";
import { CheckIcon, ChevronLeft, ChevronRight, CopyIcon } from "lucide-react";
import { SyntaxHighlighter } from "@/components/thread/syntax-highlighter";

import { TooltipIconButton } from "@/components/thread/tooltip-icon-button";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import "katex/dist/katex.min.css";

interface CodeHeaderProps {
  language?: string;
  code: string;
}

type SourceEntry = {
  label: string;
  url?: string;
  raw: string;
};

const SOURCE_URL_REGEX = /https?:\/\/[^\s\]]+/g;

function normalizeSourceEntries(sources: string[]): string[] {
  const expanded: string[] = [];
  for (const source of sources) {
    const matches = source.match(SOURCE_URL_REGEX) ?? [];
    if (matches.length > 1) {
      for (const match of matches) {
        const cleaned = match.replace(/[),.;]+$/g, "");
        if (cleaned) {
          expanded.push(cleaned);
        }
      }
      continue;
    }
    if (matches.length === 1) {
      const cleaned = matches[0].replace(/[),.;]+$/g, "");
      if (cleaned && cleaned !== matches[0]) {
        expanded.push(cleaned);
        continue;
      }
    }
    expanded.push(source);
  }
  return expanded;
}

const useCopyToClipboard = ({
  copiedDuration = 3000,
}: {
  copiedDuration?: number;
} = {}) => {
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const copyToClipboard = (value: string) => {
    if (!value) return;

    navigator.clipboard.writeText(value).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), copiedDuration);
    });
  };

  return { isCopied, copyToClipboard };
};

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const onCopy = () => {
    if (!code || isCopied) return;
    copyToClipboard(code);
  };

  return (
    <div className="flex items-center justify-between gap-4 rounded-t-lg bg-zinc-900 px-4 py-2 text-sm font-semibold text-white">
      <span className="lowercase [&>span]:text-xs">{language}</span>
      <TooltipIconButton
        tooltip="Copy"
        onClick={onCopy}
      >
        {!isCopied && <CopyIcon />}
        {isCopied && <CheckIcon />}
      </TooltipIconButton>
    </div>
  );
};

function parseSourceTag(source: string): SourceEntry {
  const trimmed = source
    .replace(/^\s*\[(?:WEB\s+)?SOURCE:\s*/i, "")
    .replace(/\]\s*$/, "")
    .trim();
  const urlMatch = trimmed.match(/https?:\/\/[^\s\]]+/);
  const rawUrl = urlMatch ? urlMatch[0] : undefined;
  const url = rawUrl ? rawUrl.replace(/[),.;]+$/g, "") : undefined;
  if (url) {
    try {
      const host = new URL(url).hostname.replace(/^www\./, "");
      return { label: host || url, url, raw: trimmed || url };
    } catch {
      return { label: url, url, raw: trimmed || url };
    }
  }
  return { label: trimmed || "Source", raw: trimmed || "Source" };
}

function SourcePill({ sources }: { sources: string[] }) {
  const parsed = normalizeSourceEntries(sources)
    .map(parseSourceTag)
    .filter((s) => s.label);
  const primary = parsed[0];
  const [activeIndex, setActiveIndex] = useState(0);
  const [isTooltipOpen, setIsTooltipOpen] = useState(false);
  if (!primary) return null;
  const hasMultiple = parsed.length > 1;

  const extraCount = parsed.length - 1;
  const activeIndexSafe =
    parsed.length > 0 ? Math.min(activeIndex, parsed.length - 1) : 0;
  const active = parsed[activeIndexSafe] ?? primary;
  const pillLabel = isTooltipOpen ? active.label : primary.label;
  const pillCount =
    !isTooltipOpen && hasMultiple && extraCount > 0 ? `+${extraCount}` : "";
  const detail =
    active.raw && active.raw !== active.label ? active.raw : active.url;

  const getFaviconUrl = (url?: string) => {
    if (!url) return undefined;
    try {
      const hostname = new URL(url).hostname;
      if (!hostname) return undefined;
      return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(
        hostname,
      )}&sz=32`;
    } catch {
      return undefined;
    }
  };

  const pillContent = (
    <span className="relative start-0 bottom-0 flex h-full w-full items-center">
      <span className="flex h-4 w-full items-center justify-between">
        <span className="max-w-[15ch] grow truncate overflow-hidden text-center text-gray-900 dark:text-zinc-100">
          {pillLabel}
        </span>
        {pillCount && (
          <span className="-me-1 flex h-full items-center rounded-full px-1 text-[#8F8F8F]">
            {pillCount}
          </span>
        )}
      </span>
    </span>
  );

  const pillShell = (
    <span data-state="closed">
      <span
        className="ms-1 inline-flex max-w-full items-center relative top-[-0.094rem] animate-[show_150ms_ease-in]"
        data-testid="webpage-citation-pill"
        style={{ width: "105px" }}
      >
        {primary.url ? (
          <a
            href={primary.url}
            target="_blank"
            rel="noopener"
            aria-label={primary.url}
            className="flex h-4.5 overflow-hidden rounded-xl px-2 text-[9px] font-medium transition-colors duration-150 ease-in-out text-gray-600 bg-[#F4F4F4] hover:bg-[#E9E9E9] dark:bg-[#303030] dark:hover:bg-[#3A3A3A]"
            style={{ maxWidth: "105px" }}
          >
            {pillContent}
          </a>
        ) : (
          <span
            className="flex h-4.5 overflow-hidden rounded-xl px-2 text-[9px] font-medium text-gray-600 bg-[#F4F4F4] dark:bg-[#303030]"
            style={{ maxWidth: "105px" }}
          >
            {pillContent}
          </span>
        )}
      </span>
    </span>
  );

  const faviconUrl = getFaviconUrl(active.url);
  const tooltipContent = (
    <div className="flex w-full items-center gap-1">
      <div className="w-full">
        <div className="w-full text-xs font-semibold whitespace-pre-wrap text-center">
          <span className="flex w-full flex-col text-start text-xs font-normal no-underline">
            <div className="flex h-9 w-full items-center justify-between gap-1.5 bg-gray-100 px-1.5 text-gray-500 dark:bg-zinc-800">
              <div className="mx-1 flex gap-1">
                <button
                  type="button"
                  aria-label="Previous source"
                  disabled={!hasMultiple}
                  className="h-6 w-6 rounded-md hover:bg-gray-200 disabled:text-gray-400 disabled:hover:bg-transparent dark:hover:bg-zinc-700"
                  onClick={() =>
                    hasMultiple &&
                    setActiveIndex(
                      (prev) => (prev - 1 + parsed.length) % parsed.length,
                    )
                  }
                >
                  <ChevronLeft className="m-auto size-4" />
                </button>
                <button
                  type="button"
                  aria-label="Next source"
                  disabled={!hasMultiple}
                  className="h-6 w-6 rounded-md hover:bg-gray-200 disabled:text-gray-400 disabled:hover:bg-transparent dark:hover:bg-zinc-700"
                  onClick={() =>
                    hasMultiple &&
                    setActiveIndex((prev) => (prev + 1) % parsed.length)
                  }
                >
                  <ChevronRight className="m-auto size-4" />
                </button>
              </div>
              <span className="mx-3.5 text-gray-500">
                {activeIndexSafe + 1}/{parsed.length}
              </span>
            </div>
            <div className="flex w-full overflow-hidden">
              <span className="flex w-full min-w-0 flex-col">
                {active.url ? (
                  <a
                    href={active.url}
                    target="_blank"
                    rel="noopener"
                    aria-label={active.url}
                    className="flex w-full flex-col gap-2 p-3"
                  >
                    <div className="flex min-w-0 w-full gap-1.5">
                      <div className="h-4 w-4 shrink-0 overflow-hidden rounded-full bg-gray-200">
                        {faviconUrl ? (
                          <img
                            alt=""
                            width={16}
                            height={16}
                            src={faviconUrl}
                            className="h-4 w-4"
                          />
                        ) : null}
                      </div>
                      <div className="min-w-0 flex-1 max-w-full truncate text-sm font-medium text-gray-900 dark:text-zinc-100">
                        {active.label}
                      </div>
                    </div>
                    {detail && (
                      <div className="line-clamp-2 break-all text-sm text-gray-600 dark:text-zinc-300">
                        {detail}
                      </div>
                    )}
                  </a>
                ) : (
                  <div className="flex w-full flex-col gap-2 p-3">
                    <div className="flex min-w-0 w-full gap-1.5">
                      <div className="h-4 w-4 shrink-0 rounded-full bg-gray-200" />
                      <div className="min-w-0 flex-1 max-w-full truncate text-sm font-medium text-gray-900 dark:text-zinc-100">
                        {active.label}
                      </div>
                    </div>
                    {detail && (
                      <div className="line-clamp-2 text-sm text-gray-600 dark:text-zinc-300">
                        {detail}
                      </div>
                    )}
                  </div>
                )}
              </span>
            </div>
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <Tooltip onOpenChange={setIsTooltipOpen}>
      <TooltipTrigger asChild>{pillShell}</TooltipTrigger>
      <TooltipContent
        side="bottom"
        align="start"
        sideOffset={6}
        hideArrow
        className="relative z-50 select-none p-0 shadow-sm rounded-xl overflow-hidden border bg-white w-80 max-w-[calc(100vw-2rem)] dark:bg-zinc-900"
      >
        {tooltipContent}
      </TooltipContent>
    </Tooltip>
  );
}

function parseSourceHref(href: string): string[] {
  if (!href.startsWith("source:")) return [];
  const encoded = href.slice("source:".length);
  try {
    const decoded = decodeURIComponent(encoded);
    const parsed = JSON.parse(decoded);
    if (Array.isArray(parsed)) return parsed.filter((s) => typeof s === "string");
    if (typeof parsed === "string") return [parsed];
    return [];
  } catch {
    return [];
  }
}

const defaultComponents: any = {
  h1: ({ className, ...props }: { className?: string }) => (
    <h1
      className={cn(
        "mb-8 scroll-m-20 text-4xl font-extrabold tracking-tight last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h2: ({ className, ...props }: { className?: string }) => (
    <h2
      className={cn(
        "mt-8 mb-4 scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h3: ({ className, ...props }: { className?: string }) => (
    <h3
      className={cn(
        "mt-6 mb-4 scroll-m-20 text-2xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h4: ({ className, ...props }: { className?: string }) => (
    <h4
      className={cn(
        "mt-6 mb-4 scroll-m-20 text-xl font-semibold tracking-tight first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h5: ({ className, ...props }: { className?: string }) => (
    <h5
      className={cn(
        "my-4 text-lg font-semibold first:mt-0 last:mb-0",
        className,
      )}
      {...props}
    />
  ),
  h6: ({ className, ...props }: { className?: string }) => (
    <h6
      className={cn("my-4 font-semibold first:mt-0 last:mb-0", className)}
      {...props}
    />
  ),
  p: ({ className, ...props }: { className?: string }) => (
    <p
      className={cn("mt-5 mb-5 leading-7 first:mt-0 last:mb-0", className)}
      {...props}
    />
  ),
  a: ({
    className,
    href,
    node,
    ...props
  }: {
    className?: string;
    href?: string;
    node?: { url?: string };
  }) => {
    const rawHref = typeof node?.url === "string" ? node.url : href;
    const normalizedHref =
      typeof rawHref === "string" && rawHref.startsWith("source%3A")
        ? decodeURIComponent(rawHref)
        : rawHref;
    if (
      typeof normalizedHref === "string" &&
      normalizedHref.startsWith("source:")
    ) {
      const sources = parseSourceHref(normalizedHref);
      return <SourcePill sources={sources} />;
    }
    return (
      <a
        className={cn(
          "text-primary font-medium underline underline-offset-4",
          className,
        )}
        href={href}
        {...props}
      />
    );
  },
  blockquote: ({ className, ...props }: { className?: string }) => (
    <blockquote
      className={cn("border-l-2 pl-6 italic", className)}
      {...props}
    />
  ),
  ul: ({ className, ...props }: { className?: string }) => (
    <ul
      className={cn("my-5 ml-6 list-disc [&>li]:mt-2", className)}
      {...props}
    />
  ),
  ol: ({ className, ...props }: { className?: string }) => (
    <ol
      className={cn("my-5 ml-6 list-decimal [&>li]:mt-2", className)}
      {...props}
    />
  ),
  hr: ({ className, ...props }: { className?: string }) => (
    <hr
      className={cn("my-5 border-b", className)}
      {...props}
    />
  ),
  table: ({ className, ...props }: { className?: string }) => (
    <div className="markdown-table-wrapper my-5 flex w-full justify-center overflow-x-auto">
      <table
        className={cn(
          "mx-auto w-fit max-w-full border-separate border-spacing-0",
          className,
        )}
        {...props}
      />
    </div>
  ),
  th: ({ className, ...props }: { className?: string }) => (
    <th
      className={cn(
        "bg-muted max-w-64 px-4 py-2 text-left align-top font-bold whitespace-normal break-words first:rounded-tl-lg last:rounded-tr-lg [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  td: ({ className, ...props }: { className?: string }) => (
    <td
      className={cn(
        "max-w-72 border-b border-l px-4 py-2 text-left align-top whitespace-normal break-all last:border-r [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  tr: ({ className, ...props }: { className?: string }) => (
    <tr
      className={cn(
        "m-0 border-b p-0 first:border-t [&:last-child>td:first-child]:rounded-bl-lg [&:last-child>td:last-child]:rounded-br-lg",
        className,
      )}
      {...props}
    />
  ),
  sup: ({ className, ...props }: { className?: string }) => (
    <sup
      className={cn("[&>a]:text-xs [&>a]:no-underline", className)}
      {...props}
    />
  ),
  pre: ({ className, ...props }: { className?: string }) => (
    <pre
      className={cn(
        "max-w-4xl overflow-x-auto rounded-lg bg-black text-white",
        className,
      )}
      {...props}
    />
  ),
  code: ({
    className,
    children,
    ...props
  }: {
    className?: string;
    children: React.ReactNode;
  }) => {
    const match = /language-(\w+)/.exec(className || "");

    if (match) {
      const language = match[1];
      const code = String(children).replace(/\n$/, "");

      return (
        <>
          <CodeHeader
            language={language}
            code={code}
          />
          <SyntaxHighlighter
            language={language}
            className={className}
          >
            {code}
          </SyntaxHighlighter>
        </>
      );
    }

    return (
      <code
        className={cn("rounded font-semibold", className)}
        {...props}
      >
        {children}
      </code>
    );
  },
};

const MarkdownTextImpl: FC<{ children: string }> = ({ children }) => {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={defaultComponents}
        urlTransform={(href) => {
          if (typeof href === "string" && href.startsWith("source:")) {
            return href;
          }
          return defaultUrlTransform(href);
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
};

export const MarkdownText = memo(MarkdownTextImpl);
