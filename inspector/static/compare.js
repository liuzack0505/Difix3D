// Compare Difix reference-selection strategies: one row per run, one column per image type, for one view.

const COLUMNS = [
  { key: "render", label: "Render (step 0)", folder: "Pred", shown: true },
  { key: "reference", label: "Reference view", folder: "Ref", shown: true },
  { key: "fixed", label: "Difix fixed", folder: "Fixed", shown: true },
  { key: "finetuned", label: "Fine-tuned model render", folder: "Pred", final: true, shown: false },
];

const state = {
  runs: [],
  views: [],
  groupSize: 0,
  viewIndex: 0,
  same: {},                       // run name -> earlier run with the identical reference, for the current view
  zoom: { scale: 1, x: 0, y: 0 }, // shared by every panel; x, y are offsets as fractions of the panel
};

const MIN_SCALE = 1;
const MAX_SCALE = 64;
const DOWNLOAD_SPACING_MS = 300;

const els = {
  runList: document.getElementById("run-list"),
  toggles: document.getElementById("column-toggles"),
  viewList: document.getElementById("view-list"),
  viewTitle: document.getElementById("view-title"),
  viewPosition: document.getElementById("view-position"),
  grid: document.getElementById("grid"),
  status: document.getElementById("status"),
  downloadAll: document.getElementById("download-all"),
  downloadSlide: document.getElementById("download-slide"),
  zoomLevel: document.getElementById("zoom-level"),
  zoomReset: document.getElementById("zoom-reset"),
};

const shownColumns = () => COLUMNS.filter((c) => c.shown);
const currentView = () => state.views[state.viewIndex];
const viewLabel = (view) => view.replace(/\.png$/, "");
const runByName = (name) => state.runs.find((r) => r.name === name);

function imageUrl(run, column, view) {
  const step = column.final ? run.final_step : run.fix_step;
  return step === null ? null : `/img/${run.name}/${step}/${column.folder}/${view}`;
}

// ---- Sidebar ----

function renderRunList() {
  els.runList.replaceChildren(...state.runs.map((run) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "run-label";
    label.textContent = run.label;
    const description = document.createElement("span");
    description.className = "run-description";
    description.textContent = run.description || run.result_dir;
    li.append(label, description);
    return li;
  }));
}

function renderToggles() {
  els.toggles.replaceChildren(...COLUMNS.map((column) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = column.shown;
    box.addEventListener("change", () => { column.shown = box.checked; renderGrid(); });
    label.append(box, column.label);
    return label;
  }));
}

function renderViewList() {
  const items = [];
  state.views.forEach((view, i) => {
    if (state.groupSize > 0 && i % state.groupSize === 0) {
      const group = document.createElement("li");
      group.className = "group";
      const last = state.views[Math.min(i + state.groupSize, state.views.length) - 1];
      group.textContent = `Trajectory ${i / state.groupSize + 1}  (${viewLabel(view)}–${viewLabel(last)})`;
      items.push(group);
    }
    const li = document.createElement("li");
    li.textContent = viewLabel(view);
    li.dataset.index = i;
    li.addEventListener("click", () => selectView(i));
    items.push(li);
  });
  els.viewList.replaceChildren(...items);
}

// ---- Grid ----

function renderGrid() {
  const view = currentView();
  const columns = shownColumns();
  els.viewTitle.textContent = view ? `View ${viewLabel(view)}` : "–";
  els.viewPosition.textContent = view ? `${state.viewIndex + 1} / ${state.views.length}` : "";
  updateDownloadButtons();
  if (!view || columns.length === 0) {
    els.grid.replaceChildren();
    els.status.hidden = false;
    els.status.textContent = view ? "Select at least one image type to show." : "No views.";
    return;
  }
  els.status.hidden = true;
  els.grid.style.setProperty("--cols", columns.length);

  const cells = [document.createElement("div")];  // empty corner above the row headers
  for (const column of columns) {
    const header = document.createElement("div");
    header.className = "column-header";
    header.textContent = column.label;
    cells.push(header);
  }
  for (const run of state.runs) {
    const header = document.createElement("div");
    header.className = "row-header";
    header.innerHTML = '<span class="run-label"></span><span class="run-description"></span>';
    header.firstChild.textContent = run.label;
    header.lastChild.textContent = run.description;
    cells.push(header);
    for (const column of columns) cells.push(makePanel(run, column, view));
  }
  els.grid.replaceChildren(...cells);
  applyZoom();
}

function makePanel(run, column, view) {
  const url = imageUrl(run, column, view);
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.dataset.run = run.name;
  panel.dataset.column = column.key;

  const label = document.createElement("div");
  label.className = "panel-label";
  const name = document.createElement("span");
  name.textContent = `${run.label} · ${column.label}`;
  label.append(name);
  if (column.key === "reference") {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.dataset.badgeFor = run.name;
    label.append(badge);
  }

  const body = document.createElement("div");
  body.className = "panel-body";
  if (url) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = `${run.label} ${column.label}, view ${viewLabel(view)}`;
    img.draggable = false;
    img.addEventListener("load", () => {
      els.grid.style.setProperty("--image-aspect", img.naturalWidth / img.naturalHeight);
      applyZoom();
    });
    img.addEventListener("error", () => body.replaceChildren(missing("Image not found")));
    body.append(img);
  } else {
    body.append(missing("No fine-tuned render in this run"));
  }
  panel.append(label, body);
  updateBadge(panel.querySelector(".badge"));
  return panel;
}

function missing(text) {
  const span = document.createElement("span");
  span.className = "missing";
  span.textContent = text;
  return span;
}

function updateBadge(badge) {
  if (!badge) return;
  const same = state.same[badge.dataset.badgeFor];
  badge.classList.toggle("same", Boolean(same));
  badge.textContent = same ? `same as ${runByName(same).label}` : "unique";
}

async function loadSameReference(view) {
  state.same = {};
  try {
    const response = await fetch(`/api/same_reference?view=${encodeURIComponent(view)}`);
    if (response.ok && view === currentView()) state.same = await response.json();
  } catch (_) { /* badges just stay "unique" */ }
  els.grid.querySelectorAll(".badge").forEach(updateBadge);
}

function selectView(index) {
  if (state.views.length === 0) return;
  state.viewIndex = Math.max(0, Math.min(index, state.views.length - 1));
  els.viewList.querySelectorAll("li[data-index]").forEach((li) => {
    li.classList.toggle("active", Number(li.dataset.index) === state.viewIndex);
  });
  els.viewList.querySelector(`li[data-index="${state.viewIndex}"]`)?.scrollIntoView({ block: "nearest" });
  renderGrid();
  loadSameReference(currentView());
  history.replaceState(null, "", `#view=${viewLabel(currentView())}`);
  preloadNeighbours();
}

function preloadNeighbours() {
  for (const offset of [1, -1]) {
    const view = state.views[state.viewIndex + offset];
    if (!view) continue;
    for (const run of state.runs) for (const column of shownColumns()) {
      const url = imageUrl(run, column, view);
      if (url) new Image().src = url;
    }
  }
}

// ---- Synchronized zoom and pan (same behaviour as the inspector) ----

function clampZoom() {
  const z = state.zoom;
  z.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, z.scale));
  z.x = Math.min(0, Math.max(1 - z.scale, z.x));
  z.y = Math.min(0, Math.max(1 - z.scale, z.y));
}

function applyZoom() {
  clampZoom();
  const { scale, x, y } = state.zoom;
  const images = els.grid.querySelectorAll(".panel-body img");
  images.forEach((img) => { img.style.transform = `translate(${x * 100}%, ${y * 100}%) scale(${scale})`; });
  els.zoomLevel.textContent = `${Math.round(scale * 100)}%`;
  const img = [...images].find((i) => i.naturalWidth > 0);
  if (img) els.grid.classList.toggle("pixelated", (img.clientWidth / img.naturalWidth) * scale * devicePixelRatio >= 2);
}

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

els.grid.addEventListener("wheel", (event) => {
  const body = event.target.closest(".panel-body");
  if (!body) return;
  event.preventDefault();
  const rect = body.getBoundingClientRect();
  zoomAt((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height,
    Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015)));
}, { passive: false });

let drag = null;
els.grid.addEventListener("pointerdown", (event) => {
  const body = event.target.closest(".panel-body");
  if (!body || event.button !== 0) return;
  body.setPointerCapture(event.pointerId);
  body.classList.add("dragging");
  drag = { body, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, x: state.zoom.x, y: state.zoom.y };
});
els.grid.addEventListener("pointermove", (event) => {
  if (!drag || event.pointerId !== drag.pointerId) return;
  const rect = drag.body.getBoundingClientRect();
  state.zoom.x = drag.x + (event.clientX - drag.startX) / rect.width;
  state.zoom.y = drag.y + (event.clientY - drag.startY) / rect.height;
  applyZoom();
});
const endDrag = (event) => {
  if (!drag || event.pointerId !== drag.pointerId) return;
  drag.body.classList.remove("dragging");
  drag = null;
};
els.grid.addEventListener("pointerup", endDrag);
els.grid.addEventListener("pointercancel", endDrag);
els.grid.addEventListener("dblclick", (event) => { if (event.target.closest(".panel-body")) resetZoom(); });
els.zoomReset.addEventListener("click", resetZoom);
window.addEventListener("resize", applyZoom);

// ---- Downloads ----

function frameImages() {
  const view = currentView();
  const items = [];
  for (const run of state.runs) for (const column of shownColumns()) {
    const url = imageUrl(run, column, view);
    if (url) items.push({ run, column, url, name: `view${viewLabel(view)}_${run.name}_${column.key}.png` });
  }
  return items;
}

function updateDownloadButtons() {
  const count = currentView() ? frameImages().length : 0;
  els.downloadAll.textContent = `Download all images (${count})`;
  els.downloadAll.disabled = count === 0;
  els.downloadSlide.disabled = count === 0;
}

// Each image is saved as its own PNG, one after another (the browser may ask once to allow multiple downloads).
function downloadAll() {
  frameImages().forEach((item, i) => {
    setTimeout(() => saveUrl(`/download?${new URLSearchParams({ f: item.url, n: item.name })}`, ""), i * DOWNLOAD_SPACING_MS);
  });
}

function saveUrl(href, filename) {
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
}

// One labelled PNG of the grid, for slides. While zoomed in, every panel shows the same zoomed region.
const SLIDE = {
  panelWidth: 960, gap: 16, margin: 40, rowHeader: 330, title: 64, columnHeader: 46,
  background: null,  // null = transparent PNG; set a colour (e.g. "#ffffff") for a solid background
  ink: "#1f1e1c", muted: "#55534d", accent: "#2a78d6", accentSoft: "#e4ecfb",
  font: 'system-ui, -apple-system, "Segoe UI", sans-serif',
};

async function downloadSlide() {
  const view = currentView();
  const columns = shownColumns();
  if (!view || columns.length === 0) return;
  els.downloadSlide.disabled = true;
  els.downloadSlide.textContent = "Rendering…";
  try {
    const bitmaps = new Map();
    await Promise.all(frameImages().map(async (item) => {
      const response = await fetch(item.url);
      if (response.ok) bitmaps.set(item.url, await createImageBitmap(await response.blob()));
    }));
    const any = bitmaps.values().next().value;
    const aspect = any ? any.width / any.height : 2.187;
    const pw = SLIDE.panelWidth, ph = Math.round(pw / aspect);
    const width = SLIDE.margin * 2 + SLIDE.rowHeader + columns.length * pw + (columns.length - 1) * SLIDE.gap;
    const height = SLIDE.margin * 2 + SLIDE.title + SLIDE.columnHeader + state.runs.length * ph + (state.runs.length - 1) * SLIDE.gap;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (SLIDE.background) {
      ctx.fillStyle = SLIDE.background;
      ctx.fillRect(0, 0, width, height);
    }

    const { scale, x, y } = state.zoom;
    const zoomed = scale > 1.001;
    ctx.fillStyle = SLIDE.ink;
    ctx.textBaseline = "top";
    ctx.font = `700 40px ${SLIDE.font}`;
    ctx.fillText(`View ${viewLabel(view)} · Difix reference selection${zoomed ? `  (zoom ${Math.round(scale * 100)}%)` : ""}`,
      SLIDE.margin, SLIDE.margin);

    const left = SLIDE.margin + SLIDE.rowHeader;
    const top = SLIDE.margin + SLIDE.title + SLIDE.columnHeader;
    ctx.font = `600 26px ${SLIDE.font}`;
    ctx.fillStyle = SLIDE.muted;
    columns.forEach((column, c) => ctx.fillText(column.label, left + c * (pw + SLIDE.gap), SLIDE.margin + SLIDE.title + 6));

    // Visible image fraction under the shared zoom: [u0, u1] x [v0, v1].
    const u0 = -x / scale, v0 = -y / scale, span = 1 / scale;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    state.runs.forEach((run, r) => {
      const rowTop = top + r * (ph + SLIDE.gap);
      ctx.fillStyle = SLIDE.ink;
      ctx.font = `700 30px ${SLIDE.font}`;
      ctx.fillText(run.label, SLIDE.margin, rowTop + ph / 2 - 30);
      ctx.fillStyle = SLIDE.muted;
      ctx.font = `400 21px ${SLIDE.font}`;
      wrapText(ctx, run.description || "", SLIDE.margin, rowTop + ph / 2 + 8, SLIDE.rowHeader - 24, 26);
      columns.forEach((column, c) => {
        const dx = left + c * (pw + SLIDE.gap);
        const bitmap = bitmaps.get(imageUrl(run, column, view));
        if (bitmap) {
          ctx.drawImage(bitmap, u0 * bitmap.width, v0 * bitmap.height, span * bitmap.width, span * bitmap.height, dx, rowTop, pw, ph);
        } else {
          ctx.fillStyle = "#ecebe7";
          ctx.fillRect(dx, rowTop, pw, ph);
        }
        if (column.key === "reference" && state.same[run.name]) drawBadge(ctx, `same as ${runByName(state.same[run.name]).label}`, dx + 14, rowTop + 14);
      });
    });

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    const href = URL.createObjectURL(blob);
    saveUrl(href, `view${viewLabel(view)}_reference_selection${zoomed ? "_zoomed" : ""}.png`);
    setTimeout(() => URL.revokeObjectURL(href), 5000);
  } finally {
    els.downloadSlide.textContent = "Download slide image";
    els.downloadSlide.disabled = false;
  }
}

function drawBadge(ctx, text, x, y) {
  ctx.font = `700 22px ${SLIDE.font}`;
  const w = ctx.measureText(text).width + 24, h = 36;
  ctx.fillStyle = SLIDE.accentSoft;
  ctx.strokeStyle = SLIDE.accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, h / 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = SLIDE.accent;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 12, y + h / 2 + 1);
  ctx.textBaseline = "top";
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
  let line = "";
  for (const word of text.split(" ")) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, y);
      line = word;
      y += lineHeight;
    } else {
      line = test;
    }
  }
  if (line) ctx.fillText(line, x, y);
}

els.downloadAll.addEventListener("click", downloadAll);
els.downloadSlide.addEventListener("click", downloadSlide);

// ---- Start ----

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  if (event.key === "ArrowDown" || event.key === "j") { selectView(state.viewIndex + 1); event.preventDefault(); }
  if (event.key === "ArrowUp" || event.key === "k") { selectView(state.viewIndex - 1); event.preventDefault(); }
  if (event.key === "+" || event.key === "=") zoomAt(0.5, 0.5, 1.25);
  if (event.key === "-" || event.key === "_") zoomAt(0.5, 0.5, 0.8);
  if (event.key === "0") resetZoom();
});

async function init() {
  const data = await (await fetch("/api/compare")).json();
  state.runs = data.runs;
  state.views = data.views;
  state.groupSize = data.group_size;
  renderRunList();
  renderToggles();
  renderViewList();
  const hashView = new URLSearchParams(location.hash.slice(1)).get("view");
  const index = hashView ? state.views.indexOf(`${hashView}.png`) : -1;
  selectView(index >= 0 ? index : 0);
}

init().catch((error) => {
  els.status.hidden = false;
  els.status.textContent = `Could not load the runs: ${error.message}`;
});
