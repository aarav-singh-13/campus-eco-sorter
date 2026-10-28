// Campus Eco Sorter - frontend logic
// API is served from the same origin as the site. To point at a separately hosted
// backend, set window.API_BASE = "https://your-backend.example.com" before this script.
const API = (window.API_BASE || "").replace(/\/$/, "");

const CAMPUS_CENTER = [29.8655, 77.8965]; // IIT Roorkee
const ICONS = { plastics: "🧴", paper: "📦", glass: "🫙", metal: "🥫", compost: "🍂", ewaste: "🔋", landfill: "🗑️" };

const state = {
  categories: {},
  allBins: [],
  file: null,
  coords: null,      // {lat, lon}
  result: null,      // last /api/predict response
  shownCategory: null,
  filter: null,      // category filter on the map (null = all)
  markers: {},
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------- Map ----------
const map = L.map("map", { scrollWheelZoom: false }).setView(CAMPUS_CENTER, 16);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const binLayer = L.layerGroup().addTo(map);
let meMarker = null;

function binColor(bin) {
  // Colour by the currently filtered category if the bin takes it, otherwise its first category.
  const cat = state.filter && bin.accepts.includes(state.filter) ? state.filter : bin.accepts[0];
  return state.categories[cat]?.color || "#475569";
}

function formatDistance(m) {
  if (m == null) return "";
  return m < 1000 ? `${m} m` : `${(m / 1000).toFixed(1)} km`;
}

function renderBins(bins) {
  binLayer.clearLayers();
  state.markers = {};
  const list = $("binList");
  list.innerHTML = "";

  if (!bins.length) {
    list.innerHTML = `<li class="empty">No bins recorded for this category yet.</li>`;
    return;
  }

  bins.forEach((b, i) => {
    const nearest = i === 0 && b.distance_m != null;
    const icon = L.divIcon({
      className: "",
      html: `<div class="pin ${b.verified ? "" : "unverified"}" style="background:${binColor(b)}"></div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });
    const accepts = b.accepts.map((c) => state.categories[c]?.name || c).join(", ");
    const popup = `<strong>${esc(b.name)}</strong><br>${esc(accepts)}${b.distance_m != null ? `<br>${formatDistance(b.distance_m)} away` : ""}${b.verified ? "" : "<br><em>Unverified sample location</em>"}
      <br><a href="https://www.google.com/maps/dir/?api=1&destination=${b.lat},${b.lon}&travelmode=walking" target="_blank" rel="noopener">Walking directions</a>`;
    const marker = L.marker([b.lat, b.lon], { icon, title: b.name }).bindPopup(popup).addTo(binLayer);
    state.markers[b.id] = marker;

    const li = document.createElement("li");
    li.className = "bin-item" + (nearest ? " nearest" : "");
    li.tabIndex = 0;
    li.innerHTML = `
      <div class="bin-name">${esc(b.name)}</div>
      <div class="bin-meta">
        <span>${esc(b.area)}</span>
        ${b.distance_m != null ? `<span>${formatDistance(b.distance_m)}</span>` : ""}
        ${nearest ? `<span class="pill near">Nearest</span>` : ""}
        ${b.verified ? "" : `<span class="pill">Unverified</span>`}
      </div>
      <div class="bin-dots">${b.accepts.map((c) => `<span title="${esc(state.categories[c]?.name || c)}" style="background:${state.categories[c]?.color || "#999"}"></span>`).join("")}</div>`;
    const focus = () => { map.setView([b.lat, b.lon], 18); marker.openPopup(); };
    li.addEventListener("click", focus);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") focus(); });
    list.appendChild(li);
  });

  const pts = bins.map((b) => [b.lat, b.lon]);
  if (state.coords) pts.push([state.coords.lat, state.coords.lon]);
  map.fitBounds(pts, { padding: [30, 30], maxZoom: 17 });
}

async function loadBins(category) {
  state.filter = category || null;
  const q = new URLSearchParams();
  if (category) q.set("category", category);
  if (state.coords) { q.set("lat", state.coords.lat); q.set("lon", state.coords.lon); }
  const res = await fetch(`${API}/api/bins?${q}`);
  const bins = await res.json();
  $("mapTitle").textContent = category ? `Bins for ${state.categories[category].name}` : "All bins on campus";
  renderFilters();
  renderBins(bins);
}

function renderFilters() {
  const wrap = $("filters");
  wrap.innerHTML = "";
  const make = (id, label, color) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.setAttribute("aria-pressed", String(state.filter === id));
    b.innerHTML = `${color ? `<span class="dot" style="background:${color}"></span>` : ""}${esc(label)}`;
    b.addEventListener("click", () => loadBins(id));
    wrap.appendChild(b);
  };
  make(null, "All");
  Object.entries(state.categories).forEach(([id, c]) => make(id, c.name, c.color));
}

// ---------- Upload ----------
function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showError("Please choose an image file.");
  if (file.size > 8 * 1024 * 1024) return showError("That image is over 8 MB. Try a smaller one.");
  showError("");
  state.file = file;
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("dropText").hidden = true;
  $("identifyBtn").disabled = false;
}

function showError(msg) {
  $("uploadError").textContent = msg;
  $("uploadError").hidden = !msg;
}

$("fileInput").addEventListener("change", (e) => setFile(e.target.files[0]));
const drop = $("dropZone");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

$("useLocation").addEventListener("change", (e) => {
  if (!e.target.checked) {
    state.coords = null;
    if (meMarker) { map.removeLayer(meMarker); meMarker = null; }
    return;
  }
  if (!navigator.geolocation) { showError("Location isn't available in this browser."); e.target.checked = false; return; }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      state.coords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      const icon = L.divIcon({ className: "", html: `<div class="me-pin"></div>`, iconSize: [16, 16], iconAnchor: [8, 8] });
      if (meMarker) map.removeLayer(meMarker);
      meMarker = L.marker([state.coords.lat, state.coords.lon], { icon, title: "You are here" }).bindPopup("You are here").addTo(map);
      loadBins(state.shownCategory || state.filter);
    },
    () => { showError("Couldn't get your location. Bins will be shown without distances."); e.target.checked = false; },
    { enableHighAccuracy: true, timeout: 10000 }
  );
});

$("identifyBtn").addEventListener("click", async () => {
  if (!state.file) return;
  const btn = $("identifyBtn");
  btn.disabled = true;
  btn.textContent = "Identifying…";
  showError("");

  const fd = new FormData();
  fd.append("file", state.file);
  if (state.coords) { fd.append("lat", state.coords.lat); fd.append("lon", state.coords.lon); }

  try {
    const res = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    state.result = data;
    showCategory(data.category, data.confidence);
    renderAlternatives(data);
  } catch (err) {
    showError(err.message || "Couldn't reach the server. Is the backend running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Identify waste";
  }
});

// ---------- Result ----------
function showCategory(catId, confidence) {
  const c = state.categories[catId];
  state.shownCategory = catId;
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;

  $("catBadge").style.background = c.color;
  $("catBadge").textContent = ICONS[catId] || "♻";
  $("catName").textContent = c.name;
  $("binLabel").textContent = c.bin_label;

  const pct = Math.round(confidence * 100);
  $("confText").textContent = `${pct}%`;
  $("confBar").style.width = `${pct}%`;
  $("confBar").style.background = c.color;

  $("tips").innerHTML = c.tips.map((t) => `<li>${esc(t)}</li>`).join("");
  $("examples").textContent = c.examples.join(", ");

  $("fbThanks").hidden = true;
  $("fbSelect").value = "";
  loadBins(catId);
}

function renderAlternatives(data) {
  const box = $("lowConf");
  box.hidden = !data.low_confidence;
  const alts = $("alternatives");
  alts.innerHTML = "";
  data.top3.slice(1).forEach((a) => {
    const c = state.categories[a.category];
    const b = document.createElement("button");
    b.className = "chip";
    b.innerHTML = `<span class="dot" style="background:${c.color}"></span>${esc(c.name)} · ${Math.round(a.confidence * 100)}%`;
    b.addEventListener("click", () => {
      box.hidden = true;
      showCategory(a.category, a.confidence);
      sendFeedback(a.category);
    });
    alts.appendChild(b);
  });
}

async function sendFeedback(correct) {
  if (!state.result || !state.file) return;
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("predicted", state.result.category);
  fd.append("correct", correct);
  try {
    const res = await fetch(`${API}/api/feedback`, { method: "POST", body: fd });
    if (res.ok) $("fbThanks").hidden = false;
  } catch { /* feedback is optional */ }
}

$("fbYes").addEventListener("click", () => state.result && sendFeedback(state.result.category));
$("fbSelect").addEventListener("change", (e) => {
  if (!e.target.value) return;
  showCategory(e.target.value, state.result ? (state.result.top3.find((t) => t.category === e.target.value)?.confidence || 0) : 0);
  sendFeedback(e.target.value);
  $("fbThanks").hidden = false;
});

// ---------- Init ----------
(async function init() {
  try {
    const [health, cats] = await Promise.all([
      fetch(`${API}/api/health`).then((r) => r.json()),
      fetch(`${API}/api/categories`).then((r) => r.json()),
    ]);
    state.categories = cats;
    $("demoBanner").hidden = !health.demo_mode;
    // Tell users which categories the model can't recognise yet (e.g. compost / e-waste).
    const known = health.predictable_categories || Object.keys(cats);
    const unknown = Object.keys(cats).filter((id) => !known.includes(id)).map((id) => cats[id].name);
    if (unknown.length) {
      $("scopeNote").textContent = `The model doesn't recognise ${unknown.join(" or ")} yet. For those, use the map filters below to find a bin.`;
      $("scopeNote").hidden = false;
    }
    const sel = $("fbSelect");
    Object.entries(cats).forEach(([id, c]) => {
      const o = document.createElement("option");
      o.value = id;
      o.textContent = c.name;
      sel.appendChild(o);
    });
    await loadBins(null);
  } catch (err) {
    showError("Couldn't reach the server. Start the backend (see README) and reload.");
  }
})();
