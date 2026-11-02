import { createServer } from "node:http";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { execFile } from "node:child_process";
import { join } from "node:path";
import { query, listSessions, getSessionMessages } from "@anthropic-ai/claude-agent-sdk";

const PORT = Number(process.env.PORT ?? 8787);
const CONFIG_DIR = join(homedir(), ".config", "claude-sdk-chat");
const CONFIG_PATH = join(CONFIG_DIR, "config.json");

const loadApiKey = async () => {
  try {
    return JSON.parse(await readFile(CONFIG_PATH, "utf8")).apiKey || process.env.ANTHROPIC_API_KEY;
  } catch {
    return process.env.ANTHROPIC_API_KEY;
  }
};

const saveApiKey = async (apiKey) => {
  await mkdir(CONFIG_DIR, { recursive: true, mode: 0o700 });
  await writeFile(CONFIG_PATH, JSON.stringify({ apiKey }), { mode: 0o600 });
};

let modelsCache;
const listModels = async () => {
  if (modelsCache) return modelsCache;
  // 一覧は SDK 内で決まり通信しないので、入力を待つだけのセッションを開いて聞く
  const q = query({ prompt: (async function* () { await new Promise(() => {}); })() });
  try {
    modelsCache = (await q.supportedModels()).map(({ value, displayName, description, supportedEffortLevels }) => ({ value, displayName, description, efforts: supportedEffortLevels ?? [] }));
  } finally {
    q.close();
  }
  return modelsCache;
};

const pickDirectory = () =>
  new Promise((resolve) => {
    execFile("osascript", ["-e", 'POSIX path of (choose folder with prompt "作業ディレクトリを選択")'], (err, stdout) =>
      // キャンセルされたら null を返す
      resolve(err ? null : stdout.trim().replace(/\/$/, "")),
    );
  });

const toolResultText = (content) =>
  (typeof content === "string" ? content : (content ?? []).map((c) => c.text ?? `[${c.type}]`).join("\n")).slice(0, 20000);

// コマンドの記録や system-reminder など、人が打っていない user の発言は出さない
const isTypedText = (text) => !/^\s*<(local-command|command-|system-reminder)/.test(text);

// 保存された会話を、画面がストリームと同じ形で描ける出来事の列にする
const replaySession = async (sessionId, dir) => {
  const events = [];
  for (const m of await getSessionMessages(sessionId, { dir })) {
    if (m.parent_tool_use_id) continue;
    const content = m.message?.content;
    const blocks = typeof content === "string" ? [{ type: "text", text: content }] : content ?? [];
    for (const b of blocks) {
      if (m.type === "user" && b.type === "text" && isTypedText(b.text)) events.push({ kind: "user", text: b.text });
      else if (m.type === "user" && b.type === "tool_result") events.push({ kind: "tool_result", id: b.tool_use_id, text: toolResultText(b.content), isError: !!b.is_error });
      else if (m.type === "assistant" && b.type === "text") events.push({ kind: "text", text: b.text });
      else if (m.type === "assistant" && b.type === "tool_use") events.push({ kind: "tool", id: b.id, name: b.name, input: b.input });
    }
  }
  return events;
};

const maskKey = (key) => (key ? `${key.slice(0, 10)}…${key.slice(-4)}` : null);

const readBody = async (req) => {
  let data = "";
  for await (const chunk of req) data += chunk;
  return JSON.parse(data);
};

const handleChat = async (req, res) => {
  const { prompt, cwd, sessionId, model, effort } = await readBody(req);
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  const apiKey = await loadApiKey();
  // サブスクの OAuth ではなく API キーで請求させるため、キーが無ければ送らない
  if (!apiKey) {
    send({ kind: "error", text: "API キーが未設定です" });
    return res.end();
  }

  const abort = new AbortController();
  res.on("close", () => abort.abort());

  try {
    for await (const msg of query({
      prompt,
      options: {
        cwd,
        resume: sessionId || undefined,
        model: model || undefined,
        effort: effort || undefined,
        // 確認を出す画面が無いので、すべてのツールを確認なしで通す
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
        abortController: abort,
        env: { ...process.env, ANTHROPIC_API_KEY: apiKey },
        // CLAUDE.md・スキル・設定を Claude Code と同じように読む
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
      },
    })) {
      // サブエージェント内の発言は親のツール結果として届くので、本文には流さない
      if (msg.parent_tool_use_id) continue;
      if (msg.type === "system" && msg.subtype === "init") {
        send({ kind: "session", sessionId: msg.session_id, model: msg.model });
      } else if (msg.type === "stream_event") {
        const ev = msg.event;
        if (ev.type === "content_block_start" && ev.content_block.type === "text") send({ kind: "text_start" });
        else if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") send({ kind: "text_delta", text: ev.delta.text });
      } else if (msg.type === "assistant") {
        for (const block of msg.message.content) {
          if (block.type === "tool_use") send({ kind: "tool", id: block.id, name: block.name, input: block.input });
        }
      } else if (msg.type === "user" && Array.isArray(msg.message.content)) {
        for (const block of msg.message.content) {
          if (block.type !== "tool_result") continue;
          send({ kind: "tool_result", id: block.tool_use_id, text: toolResultText(block.content), isError: !!block.is_error });
        }
      } else if (msg.type === "result") {
        send({
          kind: "result",
          sessionId: msg.session_id,
          cost: msg.total_cost_usd,
          durationMs: msg.duration_ms,
          inputTokens: (msg.usage?.input_tokens ?? 0) + (msg.usage?.cache_read_input_tokens ?? 0) + (msg.usage?.cache_creation_input_tokens ?? 0),
          outputTokens: msg.usage?.output_tokens ?? 0,
          isError: msg.is_error,
          errors: msg.errors,
        });
      }
    }
  } catch (e) {
    send({ kind: "error", text: String(e?.message ?? e) });
  }
  res.end();
};

const VENDOR = {
  "/vendor/marked.js": "./node_modules/marked/lib/marked.umd.js",
  "/vendor/purify.js": "./node_modules/dompurify/dist/purify.min.js",
};

const ALLOWED_HOSTS =[`127.0.0.1:${PORT}`, `localhost:${PORT}`];

// 確認なしで任意のコマンドが走るので、ほかのサイトからの POST と DNS rebinding を断る
const isSameOrigin = (req) => {
  if (!ALLOWED_HOSTS.includes(req.headers.host)) return false;
  const origin = req.headers.origin;
  return !origin || ALLOWED_HOSTS.some((h) => origin === `http://${h}`);
};

const route = async (req, res) => {
  if (!isSameOrigin(req)) return res.writeHead(403).end();
  const url = new URL(req.url, "http://localhost");
  if (req.method === "GET" && url.pathname === "/api/sessions") {
    const dir = url.searchParams.get("dir");
    const id = url.searchParams.get("id");
    const body = id
      ? await replaySession(id, dir)
      : (await listSessions({ dir, limit: 50 })).map(({ sessionId, summary, customTitle, firstPrompt, lastModified, gitBranch }) => ({
          sessionId, title: customTitle || summary || firstPrompt || "(無題)", lastModified, gitBranch,
        }));
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify(body));
  }
  if (req.method === "POST" && req.url === "/api/chat") return handleChat(req, res);
  if (req.url === "/api/models" || (req.method === "POST" && req.url === "/api/pick-dir")) {
    const body = req.url === "/api/models" ? await listModels() : { path: await pickDirectory() };
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify(body));
  }
  if (req.url === "/api/key") {
    if (req.method === "POST") await saveApiKey((await readBody(req)).apiKey?.trim() ?? "");
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({ masked: maskKey(await loadApiKey()) }));
  }
  if (req.method === "GET" && VENDOR[req.url]) {
    res.writeHead(200, { "Content-Type": "text/javascript; charset=utf-8" });
    return res.end(await readFile(new URL(VENDOR[req.url], import.meta.url)));
  }
  if (req.method === "GET" && req.url === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    return res.end(await readFile(new URL("./index.html", import.meta.url)));
  }
  res.writeHead(404).end();
};

createServer(async (req, res) => {
  try {
    await route(req, res);
  } catch (e) {
    console.error(e);
    if (!res.headersSent) res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(String(e?.message ?? e));
  }
}).listen(PORT, "127.0.0.1", () => console.log(`http://127.0.0.1:${PORT}`));
