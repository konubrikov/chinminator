/* Chinminator — логика UXP-панели Photoshop.
   Связь с локальным Python-сервером пайплайна (FastAPI, 127.0.0.1:8765).

   Используются только задокументированные API Photoshop UXP (PS 24.2+):
   • экспорт:  Document.saveAs.png(file, opts, asCopy=true) во временную папку
   • вставка:  imaging.createImageDataFromBuffer + imaging.putPixels в новый слой
   • пакет:    app.activeDocument = doc
*/
(function () {
  const uxp = require("uxp");
  const fs = uxp.storage.localFileSystem;
  const formats = uxp.storage.formats;
  const ps = require("photoshop");
  const { app, core, imaging, constants } = ps;

  const SERVER = "http://127.0.0.1:8765";
  const HEALTH_MS = 10000;

  const $ = (id) => document.getElementById(id);
  const logEl = $("log");
  function log(msg) {
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ---------- статус сервера ----------
  let busy = false;
  async function checkServer() {
    if (busy) return; // во время обработки не дёргаем сервер лишний раз
    const el = $("server-status");
    try {
      const r = await fetch(SERVER + "/health");
      const j = await r.json();
      const ok = j.zimage_backend && !j.zimage_backend.startsWith("UNAVAILABLE");
      el.textContent =
        (ok ? "● Сервер готов: " : "▲ Сервер запущен, но Z-Image недоступен: ") +
        j.zimage_backend;
      el.className = "status " + (ok ? "ok" : "warn");
    } catch (e) {
      el.textContent =
        "✕ Сервер не отвечает. Запустите scripts/start_server (порт 8765)";
      el.className = "status err";
    }
  }
  checkServer();
  setInterval(checkServer, HEALTH_MS);

  // ---------- слайдеры ----------
  $("threshold").addEventListener("input", (e) =>
    ($("v-thresh").textContent = e.target.value));
  $("strength").addEventListener("input", (e) =>
    ($("v-strength").textContent = e.target.value));

  const currentMode = () =>
    document.querySelector('input[name="mode"]:checked').value;

  function setBusy(on) {
    busy = on;
    $("btn-apply").disabled = on;
    $("btn-analyze").disabled = on;
  }

  // ---------- проверки документа ----------
  function assertSupported(doc) {
    if (!doc) throw new Error("Нет активного документа");
    if (doc.mode !== constants.DocumentMode.RGB)
      throw new Error(`«${doc.title}»: нужен режим RGB (Image > Mode > RGB Color)`);
    if (doc.bitsPerChannel !== constants.BitsPerChannelType.EIGHT)
      throw new Error(`«${doc.title}»: нужно 8 бит/канал (Image > Mode > 8 Bits/Channel)`);
  }

  // ---------- экспорт документа → PNG bytes (без потерь) ----------
  async function exportDoc(doc) {
    const tmp = await fs.getTemporaryFolder();
    const file = await tmp.createFile(`dc_export_${doc.id}.png`, { overwrite: true });
    await core.executeAsModal(
      () => doc.saveAs.png(file, { compression: 1 }, true), // asCopy: документ не меняется
      { commandName: "Export for Chinminator" });
    const buf = await file.read({ format: formats.binary });
    file.delete().catch(() => {});
    return buf;
  }

  function buildForm(bytes, mode) {
    const fd = new FormData();
    fd.append("file", new Blob([bytes], { type: "image/png" }), "photo.png");
    fd.append("mode", mode);
    fd.append("params", JSON.stringify({
      thresholds: { submental_ratio: parseFloat($("threshold").value) },
      quality: { denoise_strength: parseFloat($("strength").value) },
    }));
    return fd;
  }

  async function failText(r) {
    try { return (await r.json()).detail || r.statusText; }
    catch (_) { return r.statusText; }
  }

  // ---------- предпросмотр анализа (шаги 1–3, без Z-Image) ----------
  $("btn-analyze").addEventListener("click", async () => {
    setBusy(true);
    try {
      const doc = app.activeDocument;
      assertSupported(doc);
      const bytes = await exportDoc(doc);
      const r = await fetch(SERVER + "/analyze", {
        method: "POST", body: buildForm(bytes, currentMode()),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await failText(r)}`);
      const j = await r.json();
      if (j.analysis_preview_png_base64) {
        $("preview-img").src = "data:image/png;base64," + j.analysis_preview_png_base64;
        const { analysis_preview_png_base64, ...rest } = j;
        $("metrics").textContent = JSON.stringify(rest, null, 2);
        $("preview-box").classList.remove("hidden");
      }
      const verdict = j.decision && j.decision.should_edit
        ? `править (sev=${j.decision.severity})`
        : `пропуск — ${j.skipped_reason || "нет данных"}`;
      log(`Анализ «${doc.title}»: лиц=${j.faces_found}, ${verdict}`);
    } catch (e) {
      log("Ошибка анализа: " + e.message);
    } finally {
      setBusy(false);
    }
  });

  // ---------- применение (полный пайплайн) ----------
  async function processDoc(doc, mode, sideBySide) {
    assertSupported(doc);
    const bytes = await exportDoc(doc);
    const fmt = sideBySide ? "png" : "raw";
    const r = await fetch(`${SERVER}/process?sidebyside=${sideBySide ? 1 : 0}&format=${fmt}`, {
      method: "POST", body: buildForm(bytes, mode),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await failText(r)}`);

    if (r.headers.get("X-Pipeline-Edited") !== "1") {
      log(`«${doc.title}»: пропуск — ${r.headers.get("X-Pipeline-Reason")}`);
      return false;
    }
    const backend = r.headers.get("X-Pipeline-Backend");
    const width = parseInt(r.headers.get("X-Image-Width"), 10);
    const height = parseInt(r.headers.get("X-Image-Height"), 10);
    const body = await r.arrayBuffer();

    if (sideBySide) {
      // До|после шире исходного холста — открываем отдельным документом
      const tmp = await fs.getTemporaryFolder();
      const f = await tmp.createFile(`${doc.title.replace(/\.[^.]+$/, "")}_before_after.png`,
                                     { overwrite: true });
      await f.write(body, { format: formats.binary });
      await core.executeAsModal(() => app.open(f), { commandName: "Open Chinminator result" });
      log(`«${doc.title}»: до/после открыто новым документом (${backend})`);
      return true;
    }

    // Сырые RGB-байты → новый пиксельный слой поверх (недеструктивно)
    await core.executeAsModal(async () => {
      const layer = await doc.createPixelLayer({ name: "Chinminator Fixed" });
      const imageData = await imaging.createImageDataFromBuffer(new Uint8Array(body), {
        width, height, components: 3, chunky: true, colorSpace: "RGB",
      });
      try {
        await imaging.putPixels({
          documentID: doc.id, layerID: layer.id, imageData, replace: true,
          commandName: "Apply Chinminator",
        });
      } finally {
        imageData.dispose();
      }
    }, { commandName: "Apply Chinminator" });
    log(`«${doc.title}»: применено (${backend})`);
    return true;
  }

  $("btn-apply").addEventListener("click", async () => {
    setBusy(true);
    const mode = currentMode();
    const sbs = $("sidebyside").checked;
    try {
      if ($("batch-open").checked) {
        const docs = Array.from(app.documents); // снимок: sbs открывает новые
        let fixed = 0, failed = 0;
        for (const d of docs) {
          try {
            app.activeDocument = d;
            if (await processDoc(d, mode, sbs)) fixed++;
          } catch (e) {
            failed++;
            log(`«${d.title}»: ошибка — ${e.message}`); // не обрываем весь пакет
          }
        }
        log(`Пакет: документов ${docs.length}, исправлено ${fixed}, ошибок ${failed}`);
      } else {
        await processDoc(app.activeDocument, mode, sbs);
      }
    } catch (e) {
      log("Ошибка: " + e.message);
    } finally {
      setBusy(false);
    }
  });
})();
