import { useCallback, useRef, useState } from "react";

function parseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}

export function useSseChat(onEvent) {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(null);
  const eventHandlerRef = useRef(onEvent);
  eventHandlerRef.current = onEvent;

  const send = useCallback(async ({ question, image, sessionId }) => {
    const controller = new AbortController();
    abortRef.current = controller;
    setIsStreaming(true);
    const form = new FormData();
    form.set("question", question);
    form.set("session_id", sessionId || "");
    if (image) form.set("image", image);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        const message = await response.text();
        throw new Error(message || `请求失败：${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (!block.trim()) continue;
          const parsed = parseBlock(block);
          if (parsed) eventHandlerRef.current(parsed.event, parsed.data);
        }
        if (done) break;
      }
      if (buffer.trim()) {
        const parsed = parseBlock(buffer);
        if (parsed) eventHandlerRef.current(parsed.event, parsed.data);
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        eventHandlerRef.current("client_error", { message: error.message });
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, []);

  const stop = useCallback(() => abortRef.current?.abort(), []);
  return { send, stop, isStreaming };
}
