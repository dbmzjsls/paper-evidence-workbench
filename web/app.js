const state = {
  documents: [],
  activeDocumentId: null,
  activeTab: "screen",
  pollTimer: null,
};

const els = {
  fileInput: document.querySelector("#fileInput"),
  uploadBtn: document.querySelector("#uploadBtn"),
  rebuildBtn: document.querySelector("#rebuildBtn"),
  refreshBtn: document.querySelector("#refreshBtn"),
  jobStatus: document.querySelector("#jobStatus"),
  stats: document.querySelector("#stats"),
  documentList: document.querySelector("#documentList"),
  topicInput: document.querySelector("#topicInput"),
  limitInput: document.querySelector("#limitInput"),
  screenBtn: document.querySelector("#screenBtn"),
  queryBtn: document.querySelector("#queryBtn"),
  screenResults: document.querySelector("#screenResults"),
  answerResults: document.querySelector("#answerResults"),
  detailHint: document.querySelector("#detailHint"),
  detailContent: document.querySelector("#detailContent"),
  tabs: document.querySelectorAll(".tab"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function assetUrl(path) {
  if (!path) return "";
  const normalized = String(path).replaceAll("\\", "/");
  const marker = "/assets/";
  const index = normalized.lastIndexOf(marker);
  if (index >= 0) return `/assets/${normalized.slice(index + marker.length)}`;
  return "";
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  if (label) button.textContent = label;
}

async function refreshAll() {
  await Promise.all([loadStats(), loadDocuments()]);
}

async function loadStats() {
  const stats = await api("/stats");
  els.stats.innerHTML = [
    ["Papers", stats.document_count || 0],
    ["Chunks", stats.chunk_count || 0],
    ["Assets", stats.asset_count || 0],
    ["Index MB", stats.index_size_mb || 0],
  ]
    .map(([label, value]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
}

async function loadDocuments() {
  const data = await api("/documents?limit=500");
  state.documents = data.documents || [];
  if (!state.documents.length) {
    els.documentList.innerHTML = `<p>No papers indexed yet.</p>`;
    return;
  }
  els.documentList.innerHTML = state.documents
    .map(
      (doc) => `
        <button class="doc-item ${doc.document_id === state.activeDocumentId ? "active" : ""}"
          data-document-id="${doc.document_id}">
          <h3>${escapeHtml(doc.title || doc.filename)}</h3>
          <span>${escapeHtml(doc.filename)} · ${escapeHtml(doc.parser)}</span>
        </button>
      `,
    )
    .join("");
  els.documentList.querySelectorAll(".doc-item").forEach((button) => {
    button.addEventListener("click", () => showDocument(button.dataset.documentId));
  });
}

async function showDocument(documentId) {
  state.activeDocumentId = documentId;
  await loadDocuments();
  const detail = await api(`/documents/${documentId}`);
  const doc = detail.document;
  els.detailHint.textContent = doc.filename;
  const elements = (detail.elements || []).slice(0, 80);
  els.detailContent.innerHTML = `
    <section class="evidence-block">
      <h3>${escapeHtml(doc.title)}</h3>
      <p>${escapeHtml(doc.filename)} · ${escapeHtml(doc.file_type)} · ${escapeHtml(doc.parser)}</p>
      <p>${detail.stats.elements} elements · ${detail.stats.chunks} chunks · ${detail.stats.assets} assets</p>
    </section>
    ${elements
      .map((element) => {
        const url = assetUrl(element.asset_path);
        return `
          <section class="element-row">
            <div class="screen-meta">
              <span class="pill medium">${escapeHtml(element.type)}</span>
              <span>${element.page_idx == null ? "" : `page ${element.page_idx + 1}`}</span>
            </div>
            <p class="quote">${escapeHtml(element.text || element.caption || element.latex || element.html || "")}</p>
            ${url ? `<img class="asset-preview" src="${url}" alt="extracted asset" />` : ""}
          </section>
        `;
      })
      .join("")}
  `;
}

async function upload() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    els.jobStatus.textContent = "Choose files first.";
    return;
  }
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  setBusy(els.uploadBtn, true, "Uploading...");
  try {
    const result = await api("/documents/upload", { method: "POST", body: form });
    startPolling(result.job_id);
  } catch (error) {
    els.jobStatus.textContent = error.message;
  } finally {
    setBusy(els.uploadBtn, false, "Start ingest");
  }
}

async function rebuild() {
  setBusy(els.rebuildBtn, true, "Queued...");
  try {
    const result = await api("/documents/rebuild", { method: "POST" });
    startPolling(result.job_id);
  } catch (error) {
    els.jobStatus.textContent = error.message;
  } finally {
    setBusy(els.rebuildBtn, false, "Rebuild data/");
  }
}

function startPolling(jobId) {
  clearInterval(state.pollTimer);
  els.jobStatus.textContent = `Job ${jobId} queued`;
  state.pollTimer = setInterval(async () => {
    try {
      const job = await api(`/jobs/${jobId}`);
      const completed = (job.processed_files || 0) + (job.failed_files || 0);
      const failed = job.failed_files ? `, ${job.failed_files} failed` : "";
      els.jobStatus.textContent = `${job.status}: ${job.message || ""} (${completed}/${job.total_files}${failed})`;
      if (["completed", "failed", "partial"].includes(job.status)) {
        clearInterval(state.pollTimer);
        await refreshAll();
      }
    } catch (error) {
      els.jobStatus.textContent = `Waiting for job ${jobId}`;
    }
  }, 1600);
}

async function runScreen() {
  const topic = els.topicInput.value.trim();
  if (!topic) return;
  setBusy(els.screenBtn, true, "Screening...");
  showTab("screen");
  try {
    const report = await api("/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        limit: Number(els.limitInput.value || 10),
      }),
    });
    renderScreen(report);
  } catch (error) {
    els.screenResults.innerHTML = `<section class="screen-item"><p>${escapeHtml(error.message)}</p></section>`;
  } finally {
    setBusy(els.screenBtn, false, "Screen papers");
  }
}

function renderScreen(report) {
  const items = report.items || [];
  if (!items.length) {
    els.screenResults.innerHTML = `<section class="screen-item"><p>No matching evidence found.</p></section>`;
    return;
  }
  els.screenResults.innerHTML = items
    .map((item, index) => {
      const strength = item.relevance_score >= 0.68 ? "strong" : item.relevance_score >= 0.38 ? "medium" : "weak";
      return `
        <article class="screen-item">
          <div>
            <h3>${index + 1}. ${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.filename)}</p>
          </div>
          <div class="screen-meta">
            <span class="pill ${strength}">${Math.round(item.relevance_score * 100)}%</span>
            <span class="pill medium">${escapeHtml(item.decision)}</span>
          </div>
          <p><strong>Contribution:</strong> ${escapeHtml(item.core_contribution)}</p>
          <p><strong>Methods/Data:</strong> ${escapeHtml(item.methods_data)}</p>
          <p><strong>Findings:</strong> ${escapeHtml(item.main_findings)}</p>
          <p><strong>Limitations:</strong> ${escapeHtml(item.limitations)}</p>
          <p><strong>Ideas:</strong> ${escapeHtml((item.research_ideas || []).join(" "))}</p>
          <div class="evidence-links">
            ${(item.citations || [])
              .map(
                (citation) => `
                  <button class="cite-btn" data-citation='${escapeHtml(JSON.stringify(citation))}'>
                    ${escapeHtml(citation.citation_id)} · ${escapeHtml(citation.evidence_type)}
                  </button>
                `,
              )
              .join("")}
          </div>
        </article>
      `;
    })
    .join("");
  bindCitationButtons(els.screenResults);
}

async function runQuery() {
  const question = els.topicInput.value.trim();
  if (!question) return;
  setBusy(els.queryBtn, true, "Asking...");
  showTab("answer");
  try {
    const result = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    els.answerResults.innerHTML = `
      <section class="answer-block">${escapeHtml(result.answer || "")}</section>
      <section class="evidence-links">
        ${(result.citations || [])
          .map(
            (citation) => `
              <button class="cite-btn" data-citation='${escapeHtml(JSON.stringify(citation))}'>
                ${escapeHtml(citation.citation_id)} · ${escapeHtml(citation.evidence_type)}
              </button>
            `,
          )
          .join("")}
      </section>
    `;
    bindCitationButtons(els.answerResults);
  } catch (error) {
    els.answerResults.innerHTML = `<section class="answer-block">${escapeHtml(error.message)}</section>`;
  } finally {
    setBusy(els.queryBtn, false, "Ask corpus");
  }
}

function bindCitationButtons(root) {
  root.querySelectorAll(".cite-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const citation = JSON.parse(button.dataset.citation);
      showCitation(citation);
    });
  });
}

function showCitation(citation) {
  const url = assetUrl(citation.asset_path);
  els.detailHint.textContent = `${citation.filename}${citation.page ? ` · page ${citation.page}` : ""}`;
  els.detailContent.innerHTML = `
    <section class="evidence-block">
      <div class="screen-meta">
        <span class="pill strong">${escapeHtml(citation.citation_id)}</span>
        <span class="pill medium">${escapeHtml(citation.evidence_type)}</span>
      </div>
      <h3>${escapeHtml(citation.title)}</h3>
      <p>${escapeHtml(citation.section || citation.filename || "")}</p>
      <p class="quote">${escapeHtml(citation.quote)}</p>
      ${url ? `<img class="asset-preview" src="${url}" alt="evidence asset" />` : ""}
    </section>
  `;
}

function showTab(tabName) {
  state.activeTab = tabName;
  els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabName));
  els.screenResults.classList.toggle("hidden", tabName !== "screen");
  els.answerResults.classList.toggle("hidden", tabName !== "answer");
}

els.uploadBtn.addEventListener("click", upload);
els.rebuildBtn.addEventListener("click", rebuild);
els.refreshBtn.addEventListener("click", refreshAll);
els.screenBtn.addEventListener("click", runScreen);
els.queryBtn.addEventListener("click", runQuery);
els.tabs.forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

refreshAll().catch((error) => {
  els.jobStatus.textContent = error.message;
});
