/* SPARC browser demo: VAD + CAMPPlus + VoxLingua107 LID in ONNX Runtime Web, then SPARC calibration.
   Mirrors the Python reference pipeline exactly (see README in this folder). Audio never leaves the browser. */
(function () {
  "use strict";
  const ORT_DIST = "demo/vendor/ort/";
  const SR = 16000, MAX_SECONDS = 30;
  const MODELS = { vad: "demo/models/silero_vad.onnx", campplus: "demo/models/campplus.onnx", lid: "demo/models/lid.onnx" };

  const state = { sessions: {}, bundle: null, evaluation: null, slots: [null, null], mode: "sparc", examples: [] };
  const $ = (id) => document.getElementById(id);

  /* ---------- loading with byte progress ---------- */
  const load = { got: {}, total: {} };
  function showLoad(stage) {
    const keys = stage === 1 ? ["wasm", "vad", "campplus"] : ["lid"];
    const got = keys.reduce((s, k) => s + (load.got[k] || 0), 0), tot = keys.reduce((s, k) => s + (load.total[k] || 0), 0);
    const f = tot ? got / tot : 0, mb = (x) => (x / 1048576).toFixed(1);
    $("loadbar").style.width = (100 * f).toFixed(1) + "%";
    $("loadtext").textContent = stage === 1
      ? "Downloading the speaker model — " + mb(got) + " of " + mb(tot) + " MB"
      : "Speaker model ready. Downloading the language model for SPARC — " + mb(got) + " of " + mb(tot) + " MB";
  }
  async function fetchWithProgress(url, key, stage, expected) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load " + url);
    load.total[key] = +res.headers.get("content-length") || expected || 0;
    if (!res.body) { const b = new Uint8Array(await res.arrayBuffer()); load.got[key] = b.length; showLoad(stage); return b; }
    const reader = res.body.getReader(); const chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; load.got[key] = got; showLoad(stage); }
    const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
  }
  const EXPECTED = { vad: "silero_vad.onnx", campplus: "campplus.onnx", lid: "lid.onnx" };
  async function loadSession(name, stage) {
    const bytes = await fetchWithProgress(MODELS[name], name, stage, state.evaluation.models[EXPECTED[name]].bytes);
    state.sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }

  /* ---------- audio ---------- */
  async function decode(arrayBuffer) {
    const ctx = new OfflineAudioContext(1, SR, SR);
    const audio = await ctx.decodeAudioData(arrayBuffer);
    const n = Math.min(audio.length, MAX_SECONDS * SR), out = new Float32Array(n);
    for (let c = 0; c < audio.numberOfChannels; c++) { const d = audio.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / audio.numberOfChannels; }
    return { wave: out, truncated: audio.length > n };
  }

  /* ---------- silero VAD (same rules as silero get_speech_timestamps, max speech = inf) ---------- */
  async function vadSeconds(wave) {
    const s = state.sessions.vad, probs = [];
    let st = new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]);
    const sr = new ort.Tensor("int64", BigInt64Array.from([16000n]), []);
    let context = new Float32Array(64);
    for (let start = 0; start < wave.length; start += 512) {
      const x = new Float32Array(576); x.set(context, 0);
      x.set(wave.subarray(start, Math.min(start + 512, wave.length)), 64);
      const out = await s.run({ input: new ort.Tensor("float32", x, [1, 576]), state: st, sr });
      st = out.stateN; probs.push(out.output.data[0]); context = x.slice(512);
    }
    const win = 512, thr = 0.5, neg = Math.max(thr - 0.15, 0.01), minSpeech = SR * 0.25, pad = SR * 0.03, minSil = SR * 0.1;
    let triggered = false, cur = null, tempEnd = 0; const sp = [];
    probs.forEach((p, i) => {
      const smp = win * i;
      if (p >= thr && tempEnd) tempEnd = 0;
      if (p >= thr && !triggered) { triggered = true; cur = { start: smp }; return; }
      if (p < neg && triggered) {
        if (!tempEnd) tempEnd = smp;
        if (smp - tempEnd < minSil) return;
        cur.end = tempEnd; if (cur.end - cur.start > minSpeech) sp.push(cur);
        cur = null; tempEnd = 0; triggered = false;
      }
    });
    const n = wave.length;
    if (cur && n - cur.start > minSpeech) { cur.end = n; sp.push(cur); }
    sp.forEach((seg, i) => {
      if (i === 0) seg.start = Math.trunc(Math.max(0, seg.start - pad));
      if (i !== sp.length - 1) {
        const gap = sp[i + 1].start - seg.end;
        if (gap < 2 * pad) { seg.end += Math.trunc(Math.floor(gap / 2)); sp[i + 1].start = Math.trunc(Math.max(0, sp[i + 1].start - Math.floor(gap / 2))); }
        else { seg.end = Math.trunc(Math.min(n, seg.end + pad)); sp[i + 1].start = Math.trunc(Math.max(0, sp[i + 1].start - pad)); }
      } else seg.end = Math.trunc(Math.min(n, seg.end + pad));
    });
    return sp.reduce((a, seg) => a + seg.end - seg.start, 0) / SR;
  }

  /* ---------- encoders ---------- */
  async function embed(wave) {
    const out = await state.sessions.campplus.run({ wave: new ort.Tensor("float32", wave, [1, wave.length]) });
    return Float32Array.from(out.embedding.data);
  }
  async function lid(wave) {
    const out = await state.sessions.lid.run({ wave: new ort.Tensor("float32", wave, [1, wave.length]) });
    const p = Float64Array.from(out.probabilities.data), e = Float64Array.from(out.embedding.data);
    const ps = p.reduce((a, b) => a + b, 0); for (let i = 0; i < p.length; i++) p[i] /= ps;
    const en = Math.hypot(...e); for (let i = 0; i < e.length; i++) e[i] /= en;
    return { probs: p, emb: e };
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  /* ---------- 15 context features (delivery.metadata.pair_context) ---------- */
  function jensenShannon(l, r) {
    let a = 0, b = 0;
    for (let i = 0; i < l.length; i++) {
      const m = 0.5 * (l[i] + r[i]);
      if (l[i] > 0) a += l[i] * Math.log(Math.max(l[i], 1e-30) / Math.max(m, 1e-30));
      if (r[i] > 0) b += r[i] * Math.log(Math.max(r[i], 1e-30) / Math.max(m, 1e-30));
    }
    return 0.5 * (a + b);
  }
  function pairContext(A, B) {
    const da = A.duration, db = B.duration, va = A.vad, vb = B.vad, mn = Math.min(da, db);
    return [mn, mn / Math.max(da, db), 1 / Math.sqrt(Math.max(mn, 1e-6)), da, db, Math.max(da, db), Math.abs(da - db),
      va, vb, Math.min(va, vb), Math.max(va, vb), Math.abs(va - vb),
      dot(A.lid.probs, B.lid.probs), jensenShannon(A.lid.probs, B.lid.probs), dot(A.lid.emb, B.lid.emb)];
  }

  /* ---------- SPARC bundle (delivery.plugin.Plugin.predict) ---------- */
  function sourceLLR(b, s) { const S = b.source_affine; let v = S.bias; for (let k = 0; k < s.length; k++) v += ((s[k] - S.mean[k]) / S.scale[k]) * S.weights[k]; return v; }
  function contextLLR(b, s, q) {
    const base = sourceLLR(b, s), C = b.context_affine;
    const mean = s.reduce((a, x) => a + x, 0) / s.length;
    const sd = Math.sqrt(s.reduce((a, x) => a + (x - mean) ** 2, 0) / s.length), ptp = Math.max(...s) - Math.min(...s);
    const f = [base, ...q, sd, ptp, ...q.map((x) => base * x)];
    let v = C.bias; for (let j = 0; j < f.length; j++) v += ((f[j] - C.mean[j]) / C.scale[j]) * C.weights[j]; return v;
  }
  function residualInput(anchor, q, view) {
    const e = Math.log(Math.max(q[7], 0.01)), t = Math.log(Math.max(q[8], 0.01));
    let feats = [2 * e * t, e * e + t * t, e + t, q[12], Math.sqrt(Math.min(Math.max(q[13], 0), Math.log(2))), 1 - Math.min(Math.max(q[14], -1), 1)];
    if (view === "duration") feats = feats.slice(0, 3);
    return [anchor, ...feats, ...feats.map((x) => anchor * x)];
  }
  function sparcLLR(b, s, q) {
    const anchor = contextLLR(b, s, q), C = b.correction;
    if (C.kind === "context_affine") return anchor;
    let raw = residualInput(anchor, q, C.view);
    if (C.kind === "neural") raw = raw.concat(C.view === "full" ? q : q.slice(0, 12), s);
    const N = C.normalization;
    const x = raw.map((v, j) => (N.active[j] ? Math.min(Math.max((v - N.mean[j]) / N.scale[j], -N.clip), N.clip) : 0));
    let delta;
    if (C.kind === "quality") { const th = C.theta; delta = th[th.length - 1]; for (let j = 0; j < x.length; j++) delta += x[j] * th[j]; }
    else {
      const W0 = C.state["head.0.weight"], b0 = C.state["head.0.bias"], W3 = C.state["head.3.weight"], b3 = C.state["head.3.bias"];
      const f32 = Math.fround; delta = f32(b3[0]);
      for (let h = 0; h < W0.length; h++) {
        let z = f32(b0[h]); for (let j = 0; j < x.length; j++) z = f32(z + f32(f32(W0[h][j]) * f32(x[j])));
        delta = f32(delta + f32(f32(W3[0][h]) * Math.max(z, 0)));
      }
    }
    return anchor + delta;
  }

  /* ---------- per-recording analysis (cached per slot) ---------- */
  async function analyse(slot) {
    const r = state.slots[slot];
    if (!r.stage1) { r.stage1 = { duration: r.wave.length / SR, vad: await vadSeconds(r.wave), emb: await embed(r.wave) }; }
    if (state.sessions.lid && !r.lid) r.lid = await lid(r.wave);
    return r;
  }

  /* ---------- UI ---------- */
  function setStatus(t) { $("status").textContent = t; }
  const pct = (x) => (100 * x).toFixed(x > 0.995 || x < 0.005 ? 2 : 1) + "%";

  const NOTE = {
    raw: "How alike the two voices sound. Not a probability.",
    affine: "Chance of the same speaker, from the score alone.",
    sparc: "The same chance, also using duration, speech and language context."
  };
  function fillCol(id, num, verdict, same, frac, tick, note, ok) {
    const c = $("col-" + id);
    c.querySelector(".cnum").innerHTML = num;
    const v = c.querySelector(".cverdict"); v.textContent = verdict; v.className = "cverdict" + (same === null ? "" : same ? " same" : " diff");
    c.querySelector(".cbar i").style.width = (100 * frac).toFixed(1) + "%";
    c.querySelector(".cbar b").style.left = (100 * tick).toFixed(1) + "%";
    c.querySelector(".cnote").textContent = note;
    const bd = c.querySelector(".cbadge");
    bd.textContent = ok === null || ok === undefined ? "" : ok ? "✓ Correct" : "✗ Wrong";
    bd.className = "cbadge" + (ok === null || ok === undefined ? "" : ok ? " yes" : " no");
  }

  function resetBoard(text) {
    $("pairname").textContent = text || "Press Compare to score your two recordings";
    ["raw", "affine", "sparc"].forEach((id) => fillCol(id, "–", "", null, 0, 0.5, NOTE[id], null));
    ["delta", "ctx", "truth"].forEach((k) => { $(k).textContent = ""; });
    $("warn").hidden = true;
  }

  function render() {
    const out = state.result; if (!out) return;
    const thr = state.evaluation.raw_threshold, verdict = (s) => (s ? "Same speaker" : "Different speakers");
    const T = out.truth === undefined ? null : !!out.truth, okOf = (s) => (T === null ? null : s === T);
    $("pairname").innerHTML = out.title + (T === null ? "" : ' <span class="truthline">Ground truth: ' + (T ? "same speaker" : "different speakers") + "</span>");
    // raw cosine on a 0-100 similarity scale; the tick is the decision line
    const rawN = Math.round(out.cosine * 100);
    fillCol("raw", rawN + "<small>/100</small>", verdict(out.cosine >= thr), out.cosine >= thr, Math.min(Math.max(out.cosine, 0), 1), thr,
      NOTE.raw + " Decision line: " + Math.round(thr * 100) + ".", okOf(out.cosine >= thr));
    // calibrated columns: chance of the same speaker, tick at 50%
    const pAff = 1 / (1 + Math.exp(-out.affine));
    fillCol("affine", pct(pAff), verdict(out.affine > 0), out.affine > 0, pAff, 0.5, NOTE.affine, okOf(out.affine > 0));
    if (out.sparc === null) {
      fillCol("sparc", "…", "Loading language model", null, 0, 0.5, "SPARC appears as soon as the language model has loaded.", null);
      $("delta").textContent = "";
    } else {
      const pS = 1 / (1 + Math.exp(-out.sparc)), d = out.sparc - out.affine, sameNow = out.sparc > 0;
      fillCol("sparc", pct(pS), verdict(sameNow), sameNow, pS, 0.5, NOTE.sparc, okOf(sameNow));
      const who = sameNow ? "the same speaker" : "different speakers", surer = sameNow ? d > 0 : d < 0;
      $("delta").innerHTML = "Chance of the same speaker: " + pct(pAff) + " with affine calibration → " + pct(pS) + " with SPARC · " +
        '<span class="' + (Math.abs(d) < 0.05 ? "" : surer ? "up" : "down") + '">' +
        (Math.abs(d) < 0.05 ? "about equally sure" : surer ? "SPARC is more sure they are " + who : "SPARC is less sure they are " + who) + "</span>";
    }
    let ctx = "Speech " + out.q[7].toFixed(1) + " s / " + out.q[8].toFixed(1) + " s";
    if (out.sparc !== null) ctx += " · language similarity " + (100 * out.q[12]).toFixed(0) + "% · detected " + out.langs.map((l) => l.split(": ")[1]).join(" / ");
    $("ctx").textContent = ctx;
    $("truth").textContent = "";
    const nonEnglish = out.langs && out.langs.some((l) => !l.startsWith("en:"));
    $("warn").hidden = !nonEnglish;
    if (nonEnglish) $("warn").textContent = "Detected " + out.langs.map((l) => l.split(": ")[1]).join(" / ") +
      ". This demo was calibrated on English read speech, so the probability for this pair may not be reliable.";
  }

  async function compare(truth, title) {
    if (!state.slots[0] || !state.slots[1]) { setStatus("Add two recordings first."); return; }
    const token = (state.token = (state.token || 0) + 1);
    if (!state.sessions.campplus) setStatus("Waiting for the speaker model to finish downloading…");
    await state.ready1;
    if (token !== state.token) return;
    setStatus("Analysing…"); $("pairname").textContent = "Analysing " + (title || "") + "…";
    try {
      const a = await analyse(0), b = await analyse(1);
      if (token !== state.token) return;
      const cosine = dot(a.stage1.emb, b.stage1.emb);
      const pa = { duration: a.stage1.duration, vad: a.stage1.vad, lid: a.lid }, pb = { duration: b.stage1.duration, vad: b.stage1.vad, lid: b.lid };
      const haveLid = !!(a.lid && b.lid);
      const q = haveLid ? pairContext(pa, pb) : [0, 0, 0, 0, 0, 0, 0, pa.vad, pb.vad, 0, 0, 0, 0, 0, 0];
      const top = (l) => { let k = 0; for (let i = 1; i < l.probs.length; i++) if (l.probs[i] > l.probs[k]) k = i; return state.labels[k]; };
      state.result = { cosine, q, truth, title, langs: haveLid ? [top(a.lid), top(b.lid)] : null,
        affine: sourceLLR(state.bundle, [cosine]), sparc: haveLid ? sparcLLR(state.bundle, [cosine], q) : null };
      render(); setStatus("");
      state.debug = { cosine, q, affine: state.result.affine, sparc: state.result.sparc };
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) { setStatus("Error: " + e.message); }
  }

  async function setSlot(slot, arrayBuffer, name, url) {
    const { wave, truncated } = await decode(arrayBuffer);
    state.slots[slot] = { wave };
    $("name" + slot).textContent = name + (truncated ? " (first 30 s)" : "") + " · " + (wave.length / SR).toFixed(1) + " s";
    $("audio" + slot).src = url; $("audio" + slot).hidden = false;
  }

  async function loadExample(i) {
    const ex = state.examples[i];
    document.querySelectorAll(".ex").forEach((c, k) => c.classList.toggle("sel", k === i));
    for (const [slot, file] of [[0, ex.a], [1, ex.b]]) {
      const url = "demo/examples/" + file, buf = await (await fetch(url)).arrayBuffer();
      await setSlot(slot, buf, file.replace(".flac", ""), url);
    }
    await compare(ex.same, "Example " + (i + 1));
  }

  function exampleCards() {
    $("examples").innerHTML = state.examples.map((ex, i) =>
      `<div class="ex"><div class="exhead">Example ${i + 1}</div>` +
      `<div class="clip"><span>A</span><audio controls preload="metadata" src="demo/examples/${ex.a}"></audio></div>` +
      `<div class="clip"><span>B</span><audio controls preload="metadata" src="demo/examples/${ex.b}"></audio></div>` +
      `<button data-example="${i}">Compare this pair</button></div>`).join("");
    document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => loadExample(+b.dataset.example)));
    document.querySelectorAll("audio").forEach((a) => a.addEventListener("play", () =>
      document.querySelectorAll("audio").forEach((o) => { if (o !== a) o.pause(); })));
  }

  function metricsTable() {
    const m = state.evaluation.test_metrics, rows = [["raw", "Raw cosine", m.raw_cosine], ["affine", "Affine calibration", m.source_affine], ["sparc", "SPARC", m.sparc]];
    $("metrics").querySelector("tbody").innerHTML = rows.map(([k, label, r]) =>
      `<tr data-mode="${k}"><td>${label}</td><td>${(100 * r.balanced_accuracy).toFixed(2)}%</td><td>${r.eer_pct.toFixed(2)}%</td>` +
      `<td>${r.Cllr === null ? "—" : r.Cllr.toFixed(3)}</td><td>${r.ece === null ? "—" : (100 * r.ece).toFixed(2) + "%"}</td></tr>`).join("");
    $("ntest").textContent = state.evaluation.trials.test.toLocaleString();
  }

  async function init() {
    if (location.protocol === "file:") {
      $("loadtext").textContent = "This page has to be served over http(s), not opened as a file. In the docs folder run " +
        "\"python -m http.server 8000\" and open http://localhost:8000/demo.html — or use the published site.";
      $("loader").classList.add("alert"); return;
    }
    [state.bundle, state.evaluation, state.labels] = await Promise.all(
      ["demo/bundle.json", "demo/evaluation.json", "demo/lid_labels.json"].map((u) => fetch(u).then((r) => r.json())));
    state.examples = state.evaluation.examples;
    exampleCards(); metricsTable();
    [0, 1].forEach((slot) => $("file" + slot).addEventListener("change", async (e) => {
      const f = e.target.files[0]; if (!f) return;
      try { await setSlot(slot, await f.arrayBuffer(), f.name, URL.createObjectURL(f)); state.result = null; resetBoard(); setStatus(""); }
      catch (err) { setStatus("Could not read that audio file."); }
    }));
    $("go").addEventListener("click", () => { document.querySelectorAll(".ex").forEach((c) => c.classList.remove("sel")); compare(undefined, "Your recordings"); });

    ort.env.wasm.numThreads = 1;
    const ortBase = new URL(ORT_DIST, location.href).href;  // must be absolute: the loader resolves it relative to the script
    state.ready1 = (async () => {
      const wasm = await fetchWithProgress(ortBase + "ort-wasm-simd-threaded.wasm", "wasm", 1, 14239897);
      // Some static servers (e.g. python -m http.server) label .mjs as text/plain, which browsers refuse to import.
      // Re-wrap the loader with the right type so the page works on any host.
      const loader = await (await fetch(ortBase + "ort-wasm-simd-threaded.mjs")).text();
      const mjs = URL.createObjectURL(new Blob([loader], { type: "text/javascript" }));
      ort.env.wasm.wasmBinary = wasm; ort.env.wasm.wasmPaths = { mjs: mjs, wasm: ortBase + "ort-wasm-simd-threaded.wasm" };
      await Promise.all([loadSession("vad", 1), loadSession("campplus", 1)]);
    })();
    state.ready1.catch((e) => { $("loadtext").textContent = "Could not start the models: " + e.message; setStatus("Could not start the models: " + e.message); });
    await state.ready1;
    showLoad(2);
    state.ready2 = loadSession("lid", 2).then(() => {
      $("loadbar").style.width = "100%"; $("loadtext").textContent = "All models ready — SPARC is available.";
      setTimeout(() => $("loader").classList.add("done"), 2000);
      state.slots.forEach((r) => { if (r) delete r.lid; });
      if (state.result) compare(state.result.truth, state.result.title);
    }).catch((e) => { $("loadtext").textContent = "Language model failed to load: " + e.message; });
  }
  window.sparcDemo = { state, sourceLLR, contextLLR, sparcLLR, pairContext, vadSeconds, embed, lid, decode };
  document.addEventListener("DOMContentLoaded", () => init().catch((e) => setStatus("Error: " + e.message)));
})();
