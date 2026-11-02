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

const usageTokens = (u) => ({
  inputTokens: (u?.input_tokens ?? 0) + (u?.cache_read_input_tokens ?? 0) + (u?.cache_creation_input_tokens ?? 0),
  outputTokens: u?.output_tokens ?? 0,
});

// 誰も見ていない会話を閉じるまでの猶予。バックグラウンドのシェルや Monitor はこの間も動き続ける
const IDLE_CLOSE_MS = 30 * 60 * 1000;

const conversations = new Map();

// 1 つの会話につき query() を開いたままにし、ターンの合間もバックグラウンドの作業と通知を受け取る
class Conversation {
  constructor(id, { cwd, sessionId, model, effort, apiKey }) {
    this.id = id;
    this.sessionId = sessionId || null;
    this.model = model;
    this.effort = effort;
    this.inbox = [];
    this.wake = null;
    this.log = [];
    this.subscribers = new Set();
    this.busy = false;
    this.closed = false;
    this.q = query({
      prompt: this.input(),
      options: {
        cwd,
        resume: this.sessionId || undefined,
        model: model || undefined,
        effort: effort || undefined,
        // 確認を出す画面が無いので、すべてのツールを確認なしで通す
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
        env: { ...process.env, ANTHROPIC_API_KEY: apiKey },
        // CLAUDE.md・スキル・設定を Claude Code と同じように読む
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
      },
    });
    this.pump();
    this.scheduleIdleClose();
  }

  async *input() {
    while (!this.closed) {
      while (this.inbox.length) yield this.inbox.shift();
      await new Promise((r) => (this.wake = r));
    }
  }

  emit(ev) {
    this.log.push(ev);
    for (const send of this.subscribers) send(ev);
  }

  setBusy(busy) {
    if (this.busy === busy) return;
    this.busy = busy;
    this.emit({ kind: "busy", busy });
  }

  async send(prompt, { model, effort }) {
    if (model !== this.model) await this.q.setModel(model || undefined);
    if (effort !== this.effort) await this.q.applyFlagSettings({ effortLevel: effort || null });
    this.model = model;
    this.effort = effort;
    this.setBusy(true);
    this.inbox.push({ type: "user", message: { role: "user", content: prompt }, parent_tool_use_id: null, session_id: this.sessionId ?? "", origin: { kind: "human" } });
    this.wake?.();
  }

  subscribe(send) {
    for (const ev of this.log) send(ev);
    this.subscribers.add(send);
    clearTimeout(this.idleTimer);
    return () => {
      this.subscribers.delete(send);
      this.scheduleIdleClose();
    };
  }

  scheduleIdleClose() {
    clearTimeout(this.idleTimer);
    if (!this.subscribers.size) this.idleTimer = setTimeout(() => this.close(), IDLE_CLOSE_MS);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.idleTimer);
    this.q.close();
    this.wake?.();
    conversations.delete(this.id);
  }

  async pump() {
    try {
      for await (const msg of this.q) this.handle(msg);
    } catch (e) {
      if (!this.closed) this.emit({ kind: "error", text: String(e?.message ?? e) });
    } finally {
      this.setBusy(false);
      this.emit({ kind: "closed" });
      this.close();
    }
  }

  handle(msg) {
    if (msg.type === "system") return this.handleSystem(msg);
    // サブエージェント内の発言は親のツール結果として届くので、本文には流さない
    if (msg.parent_tool_use_id) return;
    if (msg.type === "stream_event") {
      const ev = msg.event;
      // バックグラウンドの作業の通知を受けて、送信なしに Claude が話し始めることがある
      if (ev.type === "message_start") this.setBusy(true);
      else if (ev.type === "content_block_start" && ev.content_block.type === "text") this.emit({ kind: "text_start" });
      else if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") this.emit({ kind: "text_delta", text: ev.delta.text });
    } else if (msg.type === "assistant") {
      for (const block of msg.message.content) {
        if (block.type === "tool_use") this.emit({ kind: "tool", id: block.id, name: block.name, input: block.input });
      }
    } else if (msg.type === "user" && !msg.isReplay && Array.isArray(msg.message.content)) {
      for (const block of msg.message.content) {
        if (block.type === "tool_result") this.emit({ kind: "tool_result", id: block.tool_use_id, text: toolResultText(block.content), isError: !!block.is_error });
      }
    } else if (msg.type === "result") {
      this.sessionId = msg.session_id;
      this.emit({ kind: "result", sessionId: msg.session_id, cost: msg.total_cost_usd, durationMs: msg.duration_ms, ...usageTokens(msg.usage), isError: msg.is_error, errors: msg.errors });
      this.setBusy(false);
    }
  }

  handleSystem(msg) {
    switch (msg.subtype) {
      case "init":
        this.sessionId = msg.session_id;
        return this.emit({ kind: "session", sessionId: msg.session_id, model: msg.model });
      case "task_started":
        if (msg.ambient) return;
        return this.emit({ kind: "task", id: msg.task_id, toolUseId: msg.tool_use_id, description: msg.description, taskType: msg.task_type, subagentType: msg.subagent_type });
      case "task_updated":
        return this.emit({ kind: "task_update", id: msg.task_id, status: msg.patch.status, description: msg.patch.description, error: msg.patch.error });
      case "task_progress":
        return this.emit({ kind: "task_update", id: msg.task_id, summary: msg.summary, lastTool: msg.last_tool_name, usage: msg.usage });
      case "task_notification":
        if (msg.ambient) return;
        return this.emit({ kind: "task_update", id: msg.task_id, status: msg.status, summary: msg.summary, usage: msg.usage });
      case "background_tasks_changed":
        return this.emit({ kind: "background", tasks: msg.tasks.filter((t) => !t.ambient).map((t) => ({ id: t.task_id, type: t.task_type, description: t.description })) });
    }
  }
}

const sendJson = (res, body, status = 200) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body ?? {}));
};

const handleEvents = (req, res, id) => {
  const conv = conversations.get(id);
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  const send = (ev) => res.write(`data: ${JSON.stringify(ev)}\n\n`);
  if (!conv) return send({ kind: "closed" }), res.end();
  const unsubscribe = conv.subscribe(send);
  // 中継の途中で切られないよう、無通信の間も定期的に送る
  const ping = setInterval(() => res.write(": ping\n\n"), 20000);
  res.on("close", () => { clearInterval(ping); unsubscribe(); });
};

const handleSend = async (req, res) => {
  const { conv: id, prompt, cwd, sessionId, model, effort } = await readBody(req);
  let conv = conversations.get(id);
  if (!conv) {
    const apiKey = await loadApiKey();
    // サブスクの OAuth ではなく API キーで請求させるため、キーが無ければ送らない
    if (!apiKey) return sendJson(res, { error: "API キーが未設定です" }, 400);
    conv = new Conversation(id, { cwd, sessionId, model, effort, apiKey });
    conversations.set(id, conv);
  }
  await conv.send(prompt, { model, effort });
  sendJson(res, { ok: true });
};

const handleControl = async (req, res, action) => {
  const { conv: id, taskId } = await readBody(req);
  const conv = conversations.get(id);
  if (conv && action === "interrupt") await conv.q.interrupt();
  else if (conv && action === "stop-task") await conv.q.stopTask(taskId);
  else if (conv && action === "close") conv.close();
  sendJson(res, { ok: true });
};

const VENDOR = {
  "/vendor/marked.js": "./node_modules/marked/lib/marked.umd.js",
  "/vendor/purify.js": "./node_modules/dompurify/dist/purify.min.js",
};

const ALLOWED_HOSTS = [`127.0.0.1:${PORT}`, `localhost:${PORT}`];

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
      : (await listSessions({ dir: dir || undefined, limit: 200 })).map(({ sessionId, summary, customTitle, firstPrompt, lastModified, gitBranch, cwd }) => ({
          sessionId, title: customTitle || summary || firstPrompt || "(無題)", lastModified, gitBranch, cwd,
        }));
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify(body));
  }
  if (req.method === "GET" && url.pathname === "/api/events") return handleEvents(req, res, url.searchParams.get("conv"));
  if (req.method === "POST" && url.pathname === "/api/send") return handleSend(req, res);
  if (req.method === "POST" && ["/api/interrupt", "/api/stop-task", "/api/close"].includes(url.pathname)) return handleControl(req, res, url.pathname.slice(5));
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
