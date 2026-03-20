const $ = (id) => document.getElementById(id);

const api = (path, opts) =>
  fetch(path, {
    ...opts,
    headers: {
      ...(opts.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...opts.headers,
    },
  });

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

function showJson(data) {
  $("output").textContent = JSON.stringify(data, null, 2);
}

async function refreshHealth() {
  try {
    const r = await api("/health");
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    setStatus(
      `API ok · ${j.indexed_chunks} chunk(s) indexed · LLM: ${j.has_api_key ? "configured" : "missing key"}`,
      j.has_api_key ? "ok" : undefined
    );
    if (!j.has_api_key) setStatus("Set OPENAI_API_KEY in backend .env for embeddings and chat.", "err");
  } catch (e) {
    setStatus(String(e), "err");
  }
}

$("btnIngestText").addEventListener("click", async () => {
  const title = $("docTitle").value.trim() || "untitled";
  const text = $("docText").value.trim();
  if (!text) {
    setStatus("Paste some text to index.", "err");
    return;
  }
  setStatus("Indexing…");
  try {
    const r = await api("/ingest/text", {
      method: "POST",
      body: JSON.stringify({ title, text }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    showJson(j);
    setStatus(j.message, "ok");
    await refreshHealth();
  } catch (e) {
    setStatus(String(e), "err");
  }
});

$("fileInput").addEventListener("change", async () => {
  const f = $("fileInput").files[0];
  if (!f) return;
  const title = $("docTitle").value.trim() || f.name;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("title", title);
  setStatus("Uploading & indexing…");
  try {
    const r = await api("/ingest/file", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    showJson(j);
    setStatus(j.message, "ok");
    await refreshHealth();
  } catch (e) {
    setStatus(String(e), "err");
  }
  $("fileInput").value = "";
});

$("btnQuery").addEventListener("click", async () => {
  const question = $("question").value.trim();
  if (!question) {
    setStatus("Enter a question.", "err");
    return;
  }
  setStatus("Querying…");
  try {
    const r = await api("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    showJson(j);
    setStatus("Answer ready.", "ok");
  } catch (e) {
    setStatus(String(e), "err");
  }
});

$("btnAnalyze").addEventListener("click", async () => {
  const question = $("question").value.trim() || null;
  setStatus("Analyzing…");
  try {
    const r = await api("/analyze", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    showJson(j);
    setStatus("Analysis ready.", "ok");
  } catch (e) {
    setStatus(String(e), "err");
  }
});

refreshHealth();
