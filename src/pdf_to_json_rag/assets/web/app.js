const state = {
  documents: [],
  activeDocument: null,
  uploading: false,
  asking: false,
  processingTimer: null,
};

const elements = {
  fileInput: document.querySelector("#pdf-input"),
  dropZone: document.querySelector("#drop-zone"),
  documentList: document.querySelector("#document-list"),
  documentCount: document.querySelector("#document-count"),
  refreshDocuments: document.querySelector("#refresh-documents"),
  emptyState: document.querySelector("#empty-state"),
  processingState: document.querySelector("#processing-state"),
  processingLabel: document.querySelector("#processing-label"),
  documentWorkspace: document.querySelector("#document-workspace"),
  documentKicker: document.querySelector("#document-kicker"),
  documentTitle: document.querySelector("#document-title"),
  documentSummary: document.querySelector("#document-summary"),
  replaceDocument: document.querySelector("#replace-document"),
  questionForm: document.querySelector("#question-form"),
  question: document.querySelector("#question"),
  askButton: document.querySelector("#ask-button"),
  answerArea: document.querySelector("#answer-area"),
  inspectorEmpty: document.querySelector("#inspector-empty"),
  inspectorContent: document.querySelector("#inspector-content"),
  qualityBadge: document.querySelector("#quality-badge"),
  toast: document.querySelector("#toast"),
};

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The server returned an invalid response.");
  }
  if (!response.ok || !payload.ok) {
    throw new Error(payload?.error?.message || "The operation could not be completed.");
  }
  return payload.result;
}

let toastTimer;
function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 5200);
}

function pluralizeDocuments(count) {
  return count === 1 ? "1 document" : `${count} documents`;
}

function formatPercent(value) {
  if (typeof value !== "number") return "—";
  return `${Math.round(value * 100)}%`;
}

function formatPages(start, end = start) {
  if (!start) return "—";
  return start === end ? `p. ${start}` : `pp. ${start}–${end}`;
}

function humanStatus(status) {
  return {
    ready: "Ready",
    review: "Review",
    incomplete: "Incomplete",
  }[status] || "No data";
}

function renderDocumentList() {
  elements.documentList.replaceChildren();
  elements.documentCount.textContent = pluralizeDocuments(state.documents.length);
  if (!state.documents.length) {
    elements.documentList.append(
      makeElement("p", "document-list-empty", "No documents yet. Add your first PDF to get started."),
    );
    return;
  }

  state.documents.forEach((documentItem) => {
    const button = makeElement(
      "button",
      `document-item${state.activeDocument?.doc_id === documentItem.doc_id ? " is-active" : ""}`,
    );
    button.type = "button";
    button.dataset.docId = documentItem.doc_id;
    button.setAttribute("aria-pressed", String(state.activeDocument?.doc_id === documentItem.doc_id));
    button.append(makeElement("span", "document-item-title", documentItem.label));
    const meta = makeElement("span", "document-item-meta");
    const metaText = makeElement(
      "span",
      "",
      `${documentItem.page_count || 0} pages · ${documentItem.chunk_count || 0} chunks`,
    );
    const dot = makeElement("span", `doc-state is-${documentItem.diagnostics?.status || "incomplete"}`);
    dot.title = humanStatus(documentItem.diagnostics?.status);
    meta.append(metaText, dot);
    button.append(meta);
    button.addEventListener("click", () => selectDocument(documentItem.doc_id));
    elements.documentList.append(button);
  });
}

function resetAnswer() {
  const placeholder = makeElement("div", "answer-placeholder");
  placeholder.append(makeElement("span", "answer-rule"));
  placeholder.append(makeElement("p", "", "Your answer and its sources will appear here."));
  elements.answerArea.replaceChildren(placeholder);
}

function setView(view) {
  elements.emptyState.hidden = view !== "empty";
  elements.processingState.hidden = view !== "processing";
  elements.documentWorkspace.hidden = view !== "document";
}

function updateMeter(prefix, value) {
  const valueNode = document.querySelector(`#${prefix}-value`);
  const meter = document.querySelector(`#${prefix}-meter`);
  valueNode.textContent = formatPercent(value);
  meter.style.width = `${typeof value === "number" ? Math.max(0, Math.min(100, value * 100)) : 0}%`;
}

function addSignal(list, label, value, tone = "") {
  const item = makeElement("li");
  item.append(makeElement("span", "", label), makeElement("strong", tone, value));
  list.append(item);
}

function renderInspector(documentItem) {
  const diagnostics = documentItem.diagnostics || {};
  const inspector = diagnostics.inspector || {};
  const ocr = diagnostics.ocr || {};
  const tables = diagnostics.tables || {};
  const status = diagnostics.status || "incomplete";
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  elements.qualityBadge.textContent = humanStatus(status);
  elements.qualityBadge.className = `quality-badge is-${status}`;

  document.querySelector("#metric-pages").textContent = documentItem.page_count || "—";
  document.querySelector("#metric-chunks").textContent = documentItem.chunk_count || "—";
  document.querySelector("#metric-sections").textContent = documentItem.section_count || "—";
  document.querySelector("#metric-language").textContent = documentItem.detected_language || "—";
  updateMeter("structure", documentItem.structure_confidence);
  updateMeter("layout", documentItem.layout_confidence);

  const signalList = document.querySelector("#signal-list");
  signalList.replaceChildren();
  addSignal(
    signalList,
    "Native text",
    documentItem.chunk_count ? "available" : "missing",
    documentItem.chunk_count ? "signal-ok" : "signal-warn",
  );
  addSignal(
    signalList,
    "OCR",
    ocr.used ? `${ocr.pages_processed || ocr.pages_requiring || 0} pages` : "not needed",
    ocr.used ? "signal-warn" : "signal-ok",
  );
  addSignal(
    signalList,
    "Tables",
    tables.added ? `${tables.added} added` : (tables.pages?.length ? `${tables.pages.length} pages` : "none detected"),
    tables.added || tables.pages?.length ? "" : "signal-ok",
  );
  addSignal(
    signalList,
    "Encoding",
    inspector.encoding_issues ? "review" : "clean",
    inspector.encoding_issues ? "signal-warn" : "signal-ok",
  );

  document.querySelector("#inspector-mode").textContent = inspector.effective_mode || inspector.requested_mode || "—";
  document.querySelector("#inspector-status").textContent = inspector.status || "—";
  document.querySelector("#inspector-confidence").textContent = formatPercent(inspector.confidence);
  document.querySelector("#inspector-version").textContent = inspector.version || "—";
}

function renderDocument(documentItem) {
  state.activeDocument = documentItem;
  elements.documentKicker.textContent = [documentItem.document_type, documentItem.page_count ? `${documentItem.page_count} pages` : null]
    .filter(Boolean)
    .join(" · ") || "Active document";
  elements.documentTitle.textContent = documentItem.label;
  elements.documentSummary.textContent = documentItem.summary || "This document is ready for questions.";
  renderInspector(documentItem);
  renderDocumentList();
  resetAnswer();
  setView("document");
  elements.question.focus({ preventScroll: true });
}

async function selectDocument(docId) {
  if (state.uploading || state.asking || state.activeDocument?.doc_id === docId) return;
  try {
    const documentItem = await api(`/api/documents/${encodeURIComponent(docId)}`);
    renderDocument(documentItem);
  } catch (error) {
    showToast(error.message);
  }
}

async function loadDocuments({ preserveSelection = true } = {}) {
  try {
    const activeId = preserveSelection ? state.activeDocument?.doc_id : null;
    state.documents = await api("/api/documents");
    renderDocumentList();
    const nextId = activeId && state.documents.some((item) => item.doc_id === activeId)
      ? activeId
      : state.documents[0]?.doc_id;
    if (nextId) {
      const documentItem = await api(`/api/documents/${encodeURIComponent(nextId)}`);
      renderDocument(documentItem);
    } else {
      state.activeDocument = null;
      setView("empty");
    }
  } catch (error) {
    showToast(error.message);
    if (!state.activeDocument) setView("empty");
  }
}

function startProcessing() {
  const labels = [
    "Checking the PDF structure…",
    "Extracting text and page layout…",
    "Analyzing OCR and tables…",
    "Building the local index…",
  ];
  let index = 0;
  elements.processingLabel.textContent = labels[index];
  setView("processing");
  state.processingTimer = window.setInterval(() => {
    index = Math.min(index + 1, labels.length - 1);
    elements.processingLabel.textContent = labels[index];
  }, 2200);
}

function stopProcessing() {
  window.clearInterval(state.processingTimer);
  state.processingTimer = null;
}

async function uploadPdf(file) {
  if (!file || state.uploading) return;
  if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
    showToast("Choose a PDF file.");
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    showToast("The file exceeds the 100 MB limit.");
    return;
  }

  state.uploading = true;
  startProcessing();
  try {
    const documentItem = await api("/api/documents", {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-PDF-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    state.documents = [
      documentItem,
      ...state.documents.filter((item) => item.doc_id !== documentItem.doc_id),
    ];
    renderDocument(documentItem);
    showToast("The document is ready for questions.");
  } catch (error) {
    showToast(error.message);
    setView(state.activeDocument ? "document" : "empty");
  } finally {
    stopProcessing();
    state.uploading = false;
    elements.fileInput.value = "";
  }
}

function trustCopy(trust) {
  return {
    supported: "Supported by sources",
    partial: "Partially supported",
    review: "Needs review",
    limited: "Limited sources",
  }[trust] || "Review sources";
}

function renderAnswer(result) {
  const container = makeElement("article", "answer-result");
  const top = makeElement("div", "answer-topline");
  top.append(
    makeElement("h3", "", "Answer"),
    makeElement("span", `trust-label is-${result.trust}`, trustCopy(result.trust)),
  );
  container.append(top, makeElement("div", "answer-copy", result.answer));

  if (result.evidence?.length) {
    const citations = makeElement("ol", "citation-strip");
    result.evidence.forEach((item, index) => {
      const listItem = makeElement("li");
      const link = makeElement("a", "", `[${index + 1}] ${formatPages(item.page_start, item.page_end)}`);
      link.href = `#evidence-${index + 1}`;
      link.title = item.section_title || item.chunk_id;
      listItem.append(link);
      citations.append(listItem);
    });
    container.append(citations);

    const evidenceList = makeElement("section", "evidence-list");
    evidenceList.setAttribute("aria-label", "Source evidence");
    result.evidence.forEach((item, index) => {
      const entry = makeElement("article", "evidence-item");
      entry.id = `evidence-${index + 1}`;
      const meta = makeElement("div", "evidence-meta");
      meta.append(
        makeElement("span", "", `Source ${index + 1} · ${formatPages(item.page_start, item.page_end)}`),
        makeElement("span", "", item.section_title || "Document excerpt"),
      );
      entry.append(meta, makeElement("p", "", item.sentence));
      evidenceList.append(entry);
    });
    container.append(evidenceList);
  }

  if (result.sources?.length) {
    const details = makeElement("details", "source-details");
    details.append(makeElement("summary", "", `Show retrieved chunks (${result.sources.length})`));
    const list = makeElement("div", "source-list");
    result.sources.forEach((source) => {
      const item = makeElement("article", "source-item");
      item.append(
        makeElement(
          "strong",
          "",
          `${formatPages(source.page_start, source.page_end)} · ${source.section_title || source.chunk_type}`,
        ),
        makeElement("p", "", source.excerpt),
      );
      list.append(item);
    });
    details.append(list);
    container.append(details);
  }
  elements.answerArea.replaceChildren(container);
}

async function askQuestion(event) {
  event.preventDefault();
  if (!state.activeDocument || state.asking) return;
  const query = elements.question.value.trim();
  if (!query) return;
  state.asking = true;
  elements.askButton.disabled = true;
  elements.answerArea.replaceChildren(makeElement("div", "answer-loading", "Finding an answer and checking sources…"));
  try {
    const result = await api(`/api/documents/${encodeURIComponent(state.activeDocument.doc_id)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5 }),
    });
    renderAnswer(result);
  } catch (error) {
    resetAnswer();
    showToast(error.message);
  } finally {
    state.asking = false;
    elements.askButton.disabled = false;
  }
}

elements.fileInput.addEventListener("change", () => uploadPdf(elements.fileInput.files?.[0]));
elements.replaceDocument.addEventListener("click", () => elements.fileInput.click());
elements.refreshDocuments.addEventListener("click", () => loadDocuments());
elements.questionForm.addEventListener("submit", askQuestion);
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.question.value = button.dataset.prompt;
    elements.question.focus();
  });
});

if (elements.dropZone) {
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    });
  });
  elements.dropZone.addEventListener("drop", (event) => uploadPdf(event.dataTransfer?.files?.[0]));
}

loadDocuments({ preserveSelection: false });
