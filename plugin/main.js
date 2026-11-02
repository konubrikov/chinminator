/* Double Chin Fix — логика UXP-панели Photoshop.
   Связь с локальным Python-сервером пайплайна (FastAPI, 127.0.0.1:8765). */
(async function () {
  const { app } = require("uxp").shell;
  const ps = require("photoshop");
  const core = ps.core;
  const imaging = ps.imaging;
  const constants = ps.constants;

  const SERVER = "http://127.0.0.1:8765";

  const $ = (id) => document.getElementById(id);
  const logEl = $("log");
  function log(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ---------- статус сервера ----------
  async function checkServer() {
    try {
      const r = await fetch(SERVER + "/health");
      const j = await r.json();
      const ok = j.zimage_backend && !j.zimage_backend.startsWith("UNAVAILABLE");
      $("server-status").textContent =
        (ok ? "● Сервер готов: " : "▲ Сервер запущен, но Z-Image недоступен: ") +
        j.zimage_backend;
      $("server-status").className = "status " + (ok ? "ok" : "warn");
    } catch (e) {
      $("server-status").textContent =
        "✕ Сервер не отвечает. Запустите: uvicorn app:app --port 8765";
      $("server-status").className = "status err";
    }
  }
  checkServer();
  setInterval(checkServer, 10000);

  // ---------- слайдеры ----------
  $("threshold").addEventListener("input", (e) =>
    ($("v-thresh").textContent = e.target.value));
  $("strength").addEventListener("input", (e) =>
    ($("v-strength").textContent = e.target.value));

  // ---------- экспорт активного документа → JPEG bytes ----------
  async function exportActiveDoc(quality = 92) {
    const doc = core.activeDocument;
    if (!doc) throw new Error("Нет активного документа");
    const file = await require("uxp").storage.localFs
      .getUserWorkspace({ entrypoint: constants.Entrypoints.CACHE })
      .then((ws) => ws.getTemporaryFolder())
      .then((folder) => folder.createFile("dc_export.jpg", { overwrite: true }));

    await core.executeAsModal(async () => {
      await doc.exportDocument(file, constants.ExportType.IMAGE, {
        quality,
        embedColorProfile: false,
      });
    }, { commandName: "Export for Double Chin Fix" });

    const buf = await file.read({ format: constants.Format.binary });
    return new Uint8Array(buf);
  }

  function buildForm(bytes, mode) {
    const fd = new FormData();
    fd.append("file", new Blob([bytes], { type: "image/jpeg" }), "photo.jpg");
    fd.append("mode", mode);
    const params = JSON.stringify({
      thresholds: { submental_ratio: parseFloat($("threshold").value) },
      quality: { denoise_strength: parseFloat($("strength").value) },
    });
    fd.append("params", params);
    return fd;
  }

  // ---------- предпросмотр анализа (шаги 1–3) ----------
  $("btn-analyze").addEventListener("click", async () => {
    try {
      const bytes = await exportActiveDoc();
      const r = await fetch(SERVER + "/analyze", {
        method: "POST",
        body: buildForm(bytes, document.querySelector('input[name="mode"]:checked').value),
      });
      const j = await r.json();
      if (j.analysis_preview_png_base64) {
        $("preview-img").src = "data:image/png;base64," + j.analysis_preview_png_base64;
        $("metrics").textContent = JSON.stringify(j, (k, v) =>
          k === "analysis_preview_png_base64" ? "<png>" : v, 2);
        $("preview-box").classList.remove("hidden");
      }
      log(`Анализ: лиц=${j.faces_found} решение=${JSON.stringify(j.decision)}`);
    } catch (e) { log("Ошибка анализа: " + e.message); }
  });

  // ---------- применение (полный пайплайн) ----------
  async function processOneDoc() {
    const doc = core.activeDocument;
    const bytes = await exportActiveDoc();
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const sbs = $("sidebyside").checked;
    const r = await fetch(`${SERVER}/process?sidebyside=${sbs ? 1 : 0}`, {
      method: "POST",
      body: buildForm(bytes, mode),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const edited = r.headers.get("X-Pipeline-Edited") === "1";
    const backend = r.headers.get("X-Pipeline-Backend");
    const arr = new Uint8Array(await r.arrayBuffer());

    if (!edited) {
      log(`«${doc.title}»: пропуск — ${r.headers.get("X-Pipeline-Decision")}`);
      return false;
    }
    // Вставляем результат новым слоем поверх активного (без destructive-правок)
    await core.executeAsModal(async () => {
      const layer = await doc.layers.createLayer({ name: "DoubleChin Fixed" });
      await imaging.putPixelsIntoLayer({
        documentId: doc.id,
        layerId: layer._id,
        pixels: Array.from(arr),
        format: constants.Format.BINARY,
        imageExtents: [0, 0, doc.width, doc.height],
        srcPixelSpaceBounds: { unitRulerCoordinates: {
          left: 0, top: 0, width: doc.width, height: doc.height,
          valueUnit: constants.UnitType.PIXELS } },
      });
    }, { commandName: "Apply Double Chin Fix" });
    log(`«${doc.title}»: применено (${backend})`);
    return true;
  }

  $("btn-apply").addEventListener("click", async () => {
    const btn = $("btn-apply");
    btn.disabled = true;
    try {
      if ($("batch-open").checked) {
        const ids = core.documents.map((d) => d.id);
        let n = 0;
        for (const id of ids) {
          await core.open({ _doc_id: id });
          if (await processOneDoc()) n++;
        }
        log(`Пакет: обработано документов ${ids.length}, исправлено ${n}`);
      } else {
        await processOneDoc();
      }
    } catch (e) {
      log("Ошибка: " + e.message);
    } finally {
      btn.disabled = false;
    }
  });
})();
