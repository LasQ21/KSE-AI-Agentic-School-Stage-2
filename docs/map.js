// Interactive replay map: forecast (left) against what happened (right).
// Data: geo.js (boundaries) and forecast.js (walk-forward predictions), both built by src/site_data.py.
(function () {
  var G = window.GEO, F = window.FORECAST;
  if (!G || !F) return;
  var HS = [1, 3, 6, 24], Q = F.q, N = F.n, T0 = Date.parse(F.t0);
  var NS = "http://www.w3.org/2000/svg";
  var lang = "en", L = window.I18N_DYN.en.map;

  /* decode: forecast chars -> -1 (missing) or 0..Q; outcome chars -> -1/0/1 */
  var P = {}, Y = {}, A = [];
  HS.forEach(function (h) {
    P[h] = F.p[h].map(function (s) { var a = new Int16Array(N); for (var i = 0; i < N; i++) { var c = s.charCodeAt(i); a[i] = c === 47 ? -1 : c - 48; } return a; });
    Y[h] = F.y[h].map(function (s) { var a = new Int8Array(N); for (var i = 0; i < N; i++) { var c = s.charCodeAt(i); a[i] = c === 46 ? -1 : c - 48; } return a; });
  });
  A = F.a.map(function (s) { var a = new Int8Array(N); for (var i = 0; i < N; i++) { var c = s.charCodeAt(i); a[i] = c === 46 ? -1 : c - 48; } return a; });

  var unitsByOblast = {};
  G.units.forEach(function (u, i) { (unitsByOblast[u.o] = unitsByOblast[u.o] || []).push(i); });
  var kyivUnit = -1;
  G.units.forEach(function (u, i) { if (u.en === "Kyiv city") kyivUnit = i; });
  var raionById = {}, oblastById = {};
  G.raions.forEach(function (r) { raionById[r.id] = r; });
  G.oblasts.forEach(function (o) { oblastById[o.id] = o; });

  var state = { h: 1, level: "oblast", t: F.valid[1] - 1, sel: null, timer: null };
  var $ = function (id) { return document.getElementById(id); };
  var svgF = $("map-f"), svgO = $("map-o"), slider = $("slider"), playBtn = $("play");
  var nodes = { f: {}, o: {} };      // id -> [use element, ...] by level key "lv:id"
  var kyivDots = [];

  /* ---------- value lookup ---------- */
  function unitValue(u, t, h) {
    var p = P[h][u][t];
    return { p: p < 0 ? -1 : p / Q, y: Y[h][u][t], a: A[u][t] };
  }
  function value(level, id, t, h) {
    if (level === "raion") {
      var r = raionById[id];
      return r && r.u >= 0 ? unitValue(r.u, t, h) : null;
    }
    var us = unitsByOblast[id];
    if (!us) return null;
    var miss = 1, any = -1, act = -1;
    var anyY = 0, unknownY = false, anyA = 0;
    us.forEach(function (u) {
      var v = unitValue(u, t, h);
      if (v.p >= 0) miss *= 1 - v.p;
      if (v.y === 1) anyY = 1; else if (v.y < 0) unknownY = true;
      if (v.a === 1) anyA = 1;
    });
    return { p: 1 - miss, y: anyY ? 1 : (unknownY ? -1 : 0), a: anyA };
  }

  /* ---------- colours ---------- */
  var stops = [[247, 230, 225], [238, 127, 115], [140, 26, 19]];
  function readRamp() {
    var cs = getComputedStyle(document.documentElement);
    var out = [];
    ["--ramp0", "--ramp1", "--ramp2"].forEach(function (n) {
      var v = cs.getPropertyValue(n).trim().replace("#", "");
      if (v.length === 3) v = v.replace(/./g, "$&$&");
      out.push([parseInt(v.substr(0, 2), 16), parseInt(v.substr(2, 2), 16), parseInt(v.substr(4, 2), 16)]);
    });
    if (out.every(function (c) { return !isNaN(c[0]); })) stops = out;
  }
  function ramp(p) {
    var x = Math.max(0, Math.min(1, p)) * 2, i = x >= 1 ? 1 : 0, f = x - i;
    var a = stops[i], b = stops[i + 1];
    return "rgb(" + [0, 1, 2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * f); }).join(",") + ")";
  }
  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

  /* ---------- build ---------- */
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function build() {
    var defs = el("svg", { width: 0, height: 0, "aria-hidden": "true", style: "position:absolute" }, document.body);
    var d = el("defs", {}, defs);
    var pat = el("pattern", { id: "nocov", width: 18, height: 18, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, d);
    el("rect", { width: 18, height: 18, "class": "hatch-bg" }, pat);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 18, "class": "hatch-ln" }, pat);
    G.raions.forEach(function (r) { el("path", { id: "R" + r.id, d: r.d, "fill-rule": "evenodd" }, d); });
    G.oblasts.forEach(function (o) { el("path", { id: "O" + o.id, d: o.d, "fill-rule": "evenodd" }, d); });

    [["f", svgF], ["o", svgO]].forEach(function (pair) {
      var key = pair[0], svg = pair[1];
      svg.setAttribute("viewBox", "0 0 " + G.w + " " + G.h);
      var gr = el("g", { "class": "lv-raion" }, svg);
      G.raions.forEach(function (r) {
        var u = el("use", { href: "#R" + r.id, "data-lv": "raion", "data-id": r.id }, gr);
        if (r.u < 0) { u.setAttribute("class", "nocov"); u.setAttribute("fill", "url(#nocov)"); }
        (nodes[key]["raion:" + r.id] = [u]);
      });
      var go = el("g", { "class": "lv-oblast" }, svg);
      G.oblasts.forEach(function (o) {
        var u = el("use", { href: "#O" + o.id, "data-lv": "oblast", "data-id": o.id }, go);
        if (!unitsByOblast[o.id]) { u.setAttribute("class", "nocov"); u.setAttribute("fill", "url(#nocov)"); }
        nodes[key]["oblast:" + o.id] = [u];
      });
      var gb = el("g", { "class": "borders" }, svg);
      G.oblasts.forEach(function (o) { el("use", { href: "#O" + o.id }, gb); });
      var c = el("circle", { cx: G.kyiv[0], cy: G.kyiv[1], r: 26, "data-lv": "kyiv", "data-id": "kyiv" }, svg);
      kyivDots.push([key, c]);
    });
  }

  /* ---------- names ---------- */
  function oblastName(id) {
    var us = unitsByOblast[id];
    if (us) return lang === "uk" ? G.units[us[0]].ouk : G.units[us[0]].oen;
    var o = oblastById[id];
    if (!o) return "";
    if (lang === "uk") return o.uk === "Автономна Республіка Крим" ? o.uk : o.uk === "Севастополь" ? "м. Севастополь" : o.uk + " область";
    return o.en === "Autonomous Republic of Crimea" ? o.en : o.en === "Sevastopol" ? "Sevastopol city" : o.en + " oblast";
  }
  function raionName(id) {
    var r = raionById[id];
    if (r.u >= 0) return lang === "uk" ? G.units[r.u].uk : G.units[r.u].en;
    if (lang === "uk") return /зона/.test(r.uk) ? r.uk : r.uk + " район";
    return r.en + " raion";
  }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function areaName(level, id) { return esc(level === "raion" ? raionName(id) : oblastName(id)); }

  /* ---------- rendering ---------- */
  function fmtDate(ms, withDate) {
    var o = { timeZone: "Europe/Kyiv", hour: "2-digit", minute: "2-digit", hour12: false };
    if (withDate) { o.day = "numeric"; o.month = "short"; }
    return new Intl.DateTimeFormat(lang === "uk" ? "uk-UA" : "en-GB", o).format(new Date(ms));
  }
  function currentValue(level, id) { return value(level, id, state.t, state.h); }

  function paint() {
    var h = state.h, t = state.t, lv = state.level;
    svgF.setAttribute("data-level", lv); svgO.setAttribute("data-level", lv);
    var hi = css("--ramp2"), lo = css("--ramp0");
    var ids = lv === "raion" ? G.raions.map(function (r) { return r.id; }) : G.oblasts.map(function (o) { return o.id; });
    ids.forEach(function (id) {
      var v = currentValue(lv, id);
      if (!v || v.p < 0) return;
      nodes.f[lv + ":" + id][0].setAttribute("fill", ramp(v.p));
      nodes.o[lv + ":" + id][0].setAttribute("fill", v.y === 1 ? hi : v.y === 0 ? lo : css("--nocov"));
    });
    var kv = value("raion", G.raions.filter(function (r) { return r.u === kyivUnit; })[0].id, t, h);
    kyivDots.forEach(function (kd) {
      kd[1].setAttribute("fill", kd[0] === "f" ? ramp(kv.p) : (kv.y === 1 ? hi : lo));
    });
    var start = T0 + t * 3600e3, end = start + h * 3600e3;
    $("when").innerHTML = L.made + " <b>" + fmtDate(start, true) + "</b>" +
      "<span>" + L.covers + " " + fmtDate(start, false) + "–" + fmtDate(end, false) + "</span>";
    slider.setAttribute("aria-valuetext", fmtDate(start, true));
    renderRank(); renderReadout(); renderNote();
  }

  function renderRank() {
    var lv = state.level, rows = [];
    var ids = lv === "raion" ? G.raions.filter(function (r) { return r.u >= 0; }).map(function (r) { return r.id; })
                             : Object.keys(unitsByOblast);
    ids.forEach(function (id) {
      var v = currentValue(lv, id);
      if (v && v.p >= 0) rows.push({ id: id, v: v });
    });
    rows.sort(function (a, b) { return b.v.p - a.v.p; });
    $("rank").innerHTML = rows.slice(0, 8).map(function (r) {
      var yes = r.v.y === 1;
      return '<li><button type="button" data-id="' + r.id + '"><span class="nm">' + areaName(lv, r.id) + '</span>' +
        '<span class="bar-r"><i style="--v:' + r.v.p.toFixed(3) + '"></i></span>' +
        '<span class="pc">' + Math.round(r.v.p * 100) + '%</span>' +
        '<span class="tag' + (yes ? " yes" : "") + '">' + (r.v.y < 0 ? "" : yes ? L.tagYes : L.tagNo) + '</span></button></li>';
    }).join("");
  }

  function renderReadout() {
    var lv = state.level, box = $("readout");
    var sel = state.sel;
    if (!sel || sel.level !== lv) {
      sel = state.sel = lv === "raion"
        ? { level: lv, id: G.raions.filter(function (r) { return r.u === kyivUnit; })[0].id }
        : { level: lv, id: G.units[kyivUnit].o };
    }
    document.querySelectorAll(".umap .sel").forEach(function (n) { n.classList.remove("sel"); });
    var isKyiv = (lv === "raion" && raionById[sel.id].u === kyivUnit) || (lv === "oblast" && sel.id === G.units[kyivUnit].o);
    [nodes.f, nodes.o].forEach(function (m) {
      var n = m[lv + ":" + sel.id];
      if (n && !isKyiv) { n[0].classList.add("sel"); n[0].parentNode.appendChild(n[0]); }
    });
    kyivDots.forEach(function (kd) { kd[1].classList.toggle("sel", isKyiv); });
    var name = areaName(lv, sel.id);
    var sub = lv === "raion" ? esc(oblastName(raionById[sel.id].o)) : "";
    var v = currentValue(lv, sel.id);
    if (!v || v.p < 0) {
      box.innerHTML = "<h3>" + name + "</h3>" + (sub ? '<p class="muted" style="margin:0 0 .6rem">' + sub + "</p>" : "") + '<p class="muted" style="margin:0">' + L.noCov + "</p>";
      return;
    }
    var yn = function (x) { return x < 0 ? "—" : x === 1 ? '<span class="yes">' + L.yes + "</span>" : L.no; };
    box.innerHTML = "<h3>" + name + "</h3>" + (sub ? '<p class="muted" style="margin:-.5rem 0 .8rem">' + sub + "</p>" : "") +
      "<dl><dt>" + L.chance + " " + L.nextH[state.h] + '</dt><dd class="big">' + Math.round(v.p * 100) + "%</dd>" +
      "<dt>" + L.outcome + "</dt><dd>" + yn(v.y) + "</dd>" +
      "<dt>" + L.active + "</dt><dd>" + yn(v.a) + "</dd></dl>";
  }

  function renderNote() {
    var c = F.check[state.h];
    $("mapnote").textContent = L.note(state.h, c);
  }

  /* ---------- controls ---------- */
  function setT(t) {
    var max = F.valid[state.h] - 1;
    state.t = Math.max(0, Math.min(max, t));
    slider.max = max; slider.value = state.t;
    paint();
  }
  function stop() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    playBtn.textContent = L.play; playBtn.setAttribute("aria-pressed", "false");
  }
  playBtn.addEventListener("click", function () {
    if (state.timer) { stop(); return; }
    if (state.t >= F.valid[state.h] - 1) state.t = 0;
    playBtn.textContent = L.pause; playBtn.setAttribute("aria-pressed", "true");
    state.timer = setInterval(function () {
      if (state.t >= F.valid[state.h] - 1) { stop(); return; }
      setT(state.t + 1);
    }, 320);
  });
  slider.addEventListener("input", function () { stop(); setT(parseInt(slider.value, 10)); });
  document.querySelectorAll('input[name="mh"]').forEach(function (r) {
    r.addEventListener("change", function () { state.h = parseInt(r.value, 10); setT(state.t); });
  });
  document.querySelectorAll('input[name="ml"]').forEach(function (r) {
    r.addEventListener("change", function () { state.level = r.value; state.sel = null; setT(state.t); });
  });

  function pick(target) {
    var n = target.closest && target.closest("[data-id]");
    if (!n) return;
    var id = n.getAttribute("data-id");
    if (n.getAttribute("data-lv") === "kyiv") {
      state.sel = state.level === "raion"
        ? { level: "raion", id: G.raions.filter(function (r) { return r.u === kyivUnit; })[0].id }
        : { level: "oblast", id: G.units[kyivUnit].o };
    } else if (n.tagName.toLowerCase() === "use") {
      state.sel = { level: n.getAttribute("data-lv"), id: id };
    } else return;
    renderReadout();
  }
  [svgF, svgO].forEach(function (svg) {
    svg.addEventListener("pointerover", function (e) { if (e.pointerType === "mouse") pick(e.target); });
    svg.addEventListener("click", function (e) { pick(e.target); });
  });
  $("rank").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (b) { state.sel = { level: state.level, id: b.getAttribute("data-id") }; renderReadout(); }
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { setTimeout(function () { readRamp(); paint(); }, 0); });

  /* ---------- language ---------- */
  function setLang(l) {
    lang = l;
    L = window.I18N_DYN[l].map;
    svgF.setAttribute("aria-label", L.ariaF); svgO.setAttribute("aria-label", L.ariaO);
    slider.setAttribute("aria-label", L.slider);
    playBtn.textContent = state.timer ? L.pause : L.play;
    paint();
  }

  build();
  readRamp();
  slider.max = F.valid[1] - 1; slider.value = state.t;
  window.MAP = { setLang: setLang };
  setLang("en");
})();
