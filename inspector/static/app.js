// Difix3D inspector: pick a fix step and a view, see its render / fixed / reference / alpha images side by side.

const state = {
  types: [],            // [{folder, label}] in display order
  steps: [],            // [{step, views, types}]
  step: null,           // current step object
  viewIndex: 0,
  shown: new Set(),     // folders the user wants to see
  selected: new Set(),  // folders picked for download (kept across views)
  runName: "inspector", // result folder name, used in download file names
  // Shared zoom/pan for every panel; x, y are offsets as fractions of the panel size.
  zoom: { scale: 1, x: 0, y: 0 },
};

const MIN_SCALE = 1;
const MAX_SCALE = 64;

const els = {
  resultDir: document.getElementById("result-dir"),
  stepSelect: document.getElementById("step-select"),
  toggles: document.getElementById("type-toggles"),
  viewList: document.getElementById("view-list"),
  viewTitle: document.getElementById("view-title"),
  viewPosition: document.getElementById("view-position"),
  panels: document.getElementById("panels"),
  empty: document.getElementById("empty"),
  zoomLevel: document.getElementById("zoom-level"),
  zoomReset: document.getElementById("zoom-reset"),
  download: document.getElementById("download"),
  downloadHint: document.getElementById("download-hint"),
};

// Store the hidden types, so types the page has not seen before (e.g. Fine-tuned model render) start visible.
const STORAGE_KEY = "difix3d-inspector-hidden";

function loadShown(types) {
  let hidden = [];
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved)) hidden = saved;
  } catch (_) { /* storage unavailable: show everything */ }
  return new Set(types.map((t) => t.folder).filter((f) => !hidden.includes(f)));
}

function saveShown() {
  const hidden = state.types.map((t) => t.folder).filter((f) => !state.shown.has(f));
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(hidden)); } catch (_) { /* ignore */ }
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  return { step: params.get("step"), view: params.get("view") };
}

function writeHash() {
  const view = state.step?.views[state.viewIndex];
  if (!state.step || !view) return;
  history.replaceState(null, "", `#step=${state.step.step}&view=${view.replace(/\.png$/, "")}`);
}

// Most types live in the selected step's folder; Fine-tuned model render points at the final step's Pred folder.
function imageUrl(type, view) {
  return type.step !== undefined
    ? `/img/${type.step}/${type.source}/${view}`
    : `/img/${state.step.step}/${type.folder}/${view}`;
}

function renderToggles() {
  els.toggles.replaceChildren(...state.types.map((t) => {
    const available = state.step?.types.includes(t.folder);
    const label = document.createElement("label");
    if (!available) label.className = "disabled";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.shown.has(t.folder);
    box.disabled = !available;
    box.addEventListener("change", () => {
      box.checked ? state.shown.add(t.folder) : state.shown.delete(t.folder);
      saveShown();
      renderPanels();  // also refreshes the download count, which only counts shown panels
    });
    label.append(box, t.label);
    return label;
  }));
}

function renderViewList() {
  els.viewList.replaceChildren(...state.step.views.map((view, i) => {
    const li = document.createElement("li");
    li.textContent = view.replace(/\.png$/, "");
    li.addEventListener("click", () => selectView(i));
    return li;
  }));
}

function renderPanels() {
  const view = state.step?.views[state.viewIndex];
  const folders = state.types.filter((t) => state.shown.has(t.folder) && state.step?.types.includes(t.folder));

  els.viewTitle.textContent = view ? `View ${view.replace(/\.png$/, "")}` : "–";
  els.viewPosition.textContent = view ? `${state.viewIndex + 1} / ${state.step.views.length}` : "";

  if (!view || folders.length === 0) {
    els.panels.replaceChildren();
    els.empty.hidden = false;
    els.empty.textContent = !view ? "No novel views found for this step." : "Select at least one image type to show.";
    return;
  }
  els.empty.hidden = true;
  els.panels.style.setProperty("--cols", folders.length === 1 ? 1 : folders.length <= 4 ? 2 : 3);

  els.panels.replaceChildren(...folders.map((t) => {
    const url = imageUrl(t, view);
    const panel = document.createElement("section");
    panel.className = "panel";
    panel.dataset.folder = t.folder;
    panel.classList.toggle("selected", state.selected.has(t.folder));

    const label = document.createElement("div");
    label.className = "panel-label";
    const name = document.createElement("span");
    name.textContent = t.label;
    const tag = document.createElement("span");
    tag.className = "selected-tag";
    tag.textContent = "✓ Selected";
    name.append(tag);
    const open = document.createElement("a");
    open.href = url;
    open.target = "_blank";
    open.textContent = "full size ↗";
    label.append(name, open);

    const body = document.createElement("div");
    body.className = "panel-body";
    const img = document.createElement("img");
    img.src = url;
    img.alt = `${t.label} for view ${view}`;
    img.draggable = false;
    img.addEventListener("load", () => {
      els.panels.style.setProperty("--image-aspect", img.naturalWidth / img.naturalHeight);
      applyZoom();
    });
    img.addEventListener("error", () => {
      const missing = document.createElement("span");
      missing.className = "missing";
      missing.textContent = "Image not found";
      body.replaceChildren(missing);
    });
    body.append(img);

    panel.append(label, body);
    return panel;
  }));

  applyZoom();
  updateDownload();
  preloadNeighbours(folders);
}

// ---- Selecting panels and downloading them ----

// Selected types that are shown for the current view, in display order.
function downloadTypes() {
  return state.types.filter((t) => state.selected.has(t.folder) && state.shown.has(t.folder) && state.step?.types.includes(t.folder));
}

function updateDownload() {
  const count = state.step?.views[state.viewIndex] ? downloadTypes().length : 0;
  els.download.disabled = count === 0;
  els.download.textContent = count > 1 ? `Download (${count})` : "Download";
  els.downloadHint.textContent = count === 0 ? "Click images to select" : count === 1 ? "1 image selected" : `${count} images selected`;
}

function toggleSelected(folder) {
  state.selected.has(folder) ? state.selected.delete(folder) : state.selected.add(folder);
  els.panels.querySelector(`.panel[data-folder="${folder}"]`)?.classList.toggle("selected", state.selected.has(folder));
  updateDownload();
}

function slug(text) {
  return text.toLowerCase().replace(/\(step \d+\)/, "").trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// Each selected image is saved as its own PNG, one after another. Browsers may ask once
// whether this page may download multiple files.
const DOWNLOAD_SPACING_MS = 300;

function download() {
  const view = state.step?.views[state.viewIndex];
  const types = downloadTypes();
  if (!view || types.length === 0) return;
  const base = `${state.runName}_view${view.replace(/\.png$/, "")}`;
  types.forEach((t, i) => {
    const params = new URLSearchParams({ f: imageUrl(t, view), n: `${base}_${slug(t.label)}_step${t.step ?? state.step.step}.png` });
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = `/download?${params}`;
      link.download = "";
      document.body.append(link);
      link.click();
      link.remove();
    }, i * DOWNLOAD_SPACING_MS);
  });
}

els.download.addEventListener("click", download);

// ---- Synchronized zoom and pan ----

function clampZoom() {
  const z = state.zoom;
  z.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, z.scale));
  // Keep the zoomed image covering the panel box: offsets range from (1 - scale) to 0.
  z.x = Math.min(0, Math.max(1 - z.scale, z.x));
  z.y = Math.min(0, Math.max(1 - z.scale, z.y));
}

function applyZoom() {
  clampZoom();
  const { scale, x, y } = state.zoom;
  const images = els.panels.querySelectorAll(".panel-body img");
  // translate() percentages are relative to the image box, which equals the panel box.
  images.forEach((img) => { img.style.transform = `translate(${x * 100}%, ${y * 100}%) scale(${scale})`; });
  els.zoomLevel.textContent = `${Math.round(scale * 100)}%`;

  // Once one image pixel spans 2+ screen pixels, show crisp pixels instead of smoothing them.
  const img = [...images].find((i) => i.naturalWidth > 0);
  if (img) {
    const fitted = Math.min(img.clientWidth / img.naturalWidth, img.clientHeight / img.naturalHeight);
    els.panels.classList.toggle("pixelated", fitted * scale * devicePixelRatio >= 2);
  }
}

// Zoom by `factor` keeping the point (px, py), given as fractions of the panel, fixed on screen.
function zoomAt(px, py, factor) {
  const z = state.zoom;
  const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, z.scale * factor));
  const ratio = newScale / z.scale;
  z.x = px - (px - z.x) * ratio;
  z.y = py - (py - z.y) * ratio;
  z.scale = newScale;
  applyZoom();
}

function resetZoom() {
  state.zoom = { scale: 1, x: 0, y: 0 };
  applyZoom();
}

function panelPoint(body, event) {
  const rect = body.getBoundingClientRect();
  return { px: (event.clientX - rect.left) / rect.width, py: (event.clientY - rect.top) / rect.height, rect };
}

els.panels.addEventListener("wheel", (event) => {
  const body = event.target.closest(".panel-body");
  if (!body) return;
  event.preventDefault();
  const { px, py } = panelPoint(body, event);
  // Trackpads send many small deltas, mouse wheels a few large ones; exp() handles both smoothly.
  zoomAt(px, py, Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015)));
}, { passive: false });

let drag = null;

els.panels.addEventListener("pointerdown", (event) => {
  lastPressMoved = false;
  const body = event.target.closest(".panel-body");
  if (!body || event.button !== 0) return;
  body.setPointerCapture(event.pointerId);
  body.classList.add("dragging");
  drag = { body, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, x: state.zoom.x, y: state.zoom.y };
});

els.panels.addEventListener("pointermove", (event) => {
  if (!drag || event.pointerId !== drag.pointerId) return;
  const rect = drag.body.getBoundingClientRect();
  state.zoom.x = drag.x + (event.clientX - drag.startX) / rect.width;
  state.zoom.y = drag.y + (event.clientY - drag.startY) / rect.height;
  applyZoom();
});

// A press that moved less than this many pixels is a click (select), not a pan.
const CLICK_SLOP = 4;
let lastPressMoved = false;
let pendingSelect = null;

function endDrag(event) {
  if (!drag || event.pointerId !== drag.pointerId) return;
  lastPressMoved = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > CLICK_SLOP;
  drag.body.classList.remove("dragging");
  drag = null;
}
els.panels.addEventListener("pointerup", endDrag);
els.panels.addEventListener("pointercancel", endDrag);

// Single click toggles the panel's selection. It waits for the double-click window so a
// double-click only resets zoom and does not also toggle the selection.
els.panels.addEventListener("click", (event) => {
  const panel = event.target.closest(".panel");
  if (!panel || event.target.closest("a") || lastPressMoved || event.detail > 1) return;
  clearTimeout(pendingSelect);
  pendingSelect = setTimeout(() => toggleSelected(panel.dataset.folder), 250);
});

els.panels.addEventListener("dblclick", (event) => {
  clearTimeout(pendingSelect);
  if (event.target.closest(".panel-body")) resetZoom();
});

els.zoomReset.addEventListener("click", resetZoom);
window.addEventListener("resize", applyZoom);

// Fetch the next and previous views in the background so arrow-key browsing feels instant.
function preloadNeighbours(folders) {
  for (const offset of [1, -1]) {
    const view = state.step.views[state.viewIndex + offset];
    if (view) folders.forEach((t) => { new Image().src = imageUrl(t, view); });
  }
}

function selectView(index) {
  if (!state.step || state.step.views.length === 0) return;
  state.viewIndex = Math.max(0, Math.min(index, state.step.views.length - 1));
  [...els.viewList.children].forEach((li, i) => li.classList.toggle("active", i === state.viewIndex));
  els.viewList.children[state.viewIndex]?.scrollIntoView({ block: "nearest" });
  renderPanels();
  writeHash();
}

function selectStep(stepValue, viewName) {
  state.step = state.steps.find((s) => String(s.step) === String(stepValue)) ?? state.steps[0];
  els.stepSelect.value = String(state.step.step);
  renderToggles();
  renderViewList();
  // Keep the same view when switching steps, if the new step has it.
  const index = viewName ? state.step.views.indexOf(viewName) : -1;
  selectView(index >= 0 ? index : 0);
}

async function init() {
  const response = await fetch("/api/steps");
  const data = await response.json();
  state.types = data.types;
  state.steps = data.steps;
  state.shown = loadShown(state.types);
  state.runName = data.result_dir.split("/").filter(Boolean).pop() || "inspector";
  els.resultDir.textContent = data.result_dir;

  if (state.steps.length === 0) {
    els.empty.hidden = false;
    els.empty.textContent = "No fix steps found under renders/novel/.";
    return;
  }

  els.stepSelect.replaceChildren(...state.steps.map((s) => new Option(`Step ${s.step} (${s.views.length} views)`, s.step)));
  els.stepSelect.addEventListener("change", () => selectStep(els.stepSelect.value, state.step?.views[state.viewIndex]));

  const hash = readHash();
  selectStep(hash.step ?? state.steps[state.steps.length - 1].step, hash.view ? `${hash.view}.png` : null);
}

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "SELECT" || event.target.tagName === "INPUT") return;
  if (event.key === "ArrowDown" || event.key === "j") { selectView(state.viewIndex + 1); event.preventDefault(); }
  if (event.key === "ArrowUp" || event.key === "k") { selectView(state.viewIndex - 1); event.preventDefault(); }
  if (event.key === "+" || event.key === "=") zoomAt(0.5, 0.5, 1.25);
  if (event.key === "-" || event.key === "_") zoomAt(0.5, 0.5, 0.8);
  if (event.key === "0") resetZoom();
});

init().catch((error) => {
  els.empty.hidden = false;
  els.empty.textContent = `Could not load the result folder: ${error.message}`;
});
