import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { query } from "@anthropic-ai/claude-agent-sdk";

const PORT = Number(process.env.PORT ?? 8787);

// サブスクの OAuth ではなく API キーで請求させるため、キーが無ければ起動しない
if (!process.env.ANTHROPIC_API_KEY) {
  console.error("ANTHROPIC_API_KEY を設定してから起動してください");
  process.exit(1);
}

const readBody = async (req) => {
  let data = "";
  for await (const chunk of req) data += chunk;
  return JSON.parse(data);
};

const handleChat = async (req, res) => {
  const { prompt, cwd, sessionId, model } = await readBody(req);
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  const abort = new AbortController();
  res.on("close", () => abort.abort());

  try {
    for await (const msg of query({
      prompt,
      options: {
        cwd,
        resume: sessionId || undefined,
        model: model || undefined,
        // 確認を出す画面が無いので、すべてのツールを確認なしで通す
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
        abortController: abort,
        // CLAUDE.md・スキル・設定を Claude Code と同じように読む
        settingSources: ["user", "project", "local"],
      },
    })) {
      if (msg.type === "system" && msg.subtype === "init") {
        send({ kind: "session", sessionId: msg.session_id, model: msg.model });
      } else if (msg.type === "assistant") {
        for (const block of msg.message.content) {
          if (block.type === "text") send({ kind: "text", text: block.text });
          else if (block.type === "tool_use") send({ kind: "tool", name: block.name, input: block.input });
        }
      } else if (msg.type === "result") {
        send({ kind: "result", sessionId: msg.session_id, cost: msg.total_cost_usd, isError: msg.is_error });
      }
    }
  } catch (e) {
    send({ kind: "error", text: String(e?.message ?? e) });
  }
  res.end();
};

createServer(async (req, res) => {
  if (req.method === "POST" && req.url === "/api/chat") return handleChat(req, res);
  if (req.method === "GET" && req.url === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    return res.end(await readFile(new URL("./index.html", import.meta.url)));
  }
  res.writeHead(404).end();
}).listen(PORT, "127.0.0.1", () => console.log(`http://127.0.0.1:${PORT}`));
