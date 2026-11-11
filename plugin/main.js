// Photoshop resolves recorded handlers in the plugin global scope.
var chinminatorPlayback;
async function chinminatorAction(executionContext, info) {
  return await chinminatorPlayback(executionContext, info);
}
window.chinminatorAction = chinminatorAction;
/* Chinminator — логика UXP-панели Photoshop.
   Связь с локальным Python-сервером пайплайна (FastAPI, 127.0.0.1:8765).

   Используются только задокументированные API Photoshop UXP (PS 24.2+):
   • чтение композита: imaging.getPixels без Save As
   • вставка:  imaging.createImageDataFromBuffer + imaging.putPixels в новый слой
   • пакет:    app.activeDocument = doc
*/
(function () {
  const uxp = require("uxp");
  const fs = uxp.storage.localFileSystem;
  const formats = uxp.storage.formats;
  const ps = require("photoshop");
  const { app, core, imaging, constants } = ps;

  const SERVER = "http://localhost:8765";
  const HEALTH_MS = 10000;

  const $ = (id) => document.getElementById(id);
  const logEl = $("log");
  function log(msg) {
    $("task-status").textContent = msg;
    logEl.textContent += msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  let activePrompt = "";
  let defaultPrompt = "";
  let activeEngine = "qwen2";
  let modelLabel = "Qwen-Image 2.1";
  let fixedSteps = null;
  let syncedEngine = null;
  const promptFile = () => activeEngine === "flux3" ? "flux3-retouch-prompt.txt" : "retouch-prompt.txt";
  const promptReady = (async () => {
    const root = await fs.getPluginFolder();
    defaultPrompt = (await (await root.getEntry("default-prompt.txt")).read()).trim();
    const data = await fs.getDataFolder();
    const saved = (await data.getEntries()).find(entry => entry.name === "retouch-prompt.txt");
    activePrompt = saved ? (await saved.read()).trim() : defaultPrompt;
    if (!activePrompt) activePrompt = defaultPrompt;
    return true;
  })().catch(e => { log("Не удалось загрузить промпт: " + e.message); return false; });
  async function requirePrompt() {
    if (!(await promptReady)) throw new Error("Промпт недоступен; перезагрузите плагин");
  }
  $("btn-prompt").addEventListener("click", async () => {
    try {
      await requirePrompt();
      $("prompt-text").value = activePrompt;
      $("prompt-error").textContent = "";
      await $("prompt-dialog").uxpShowModal({ title: "Промпт ретуши", resize: "none", size: { width: 560, height: 390 } });
    } catch (e) { log("Ошибка окна промпта: " + e.message); }
  });
  $("prompt-reset").addEventListener("click", () => { $("prompt-text").value = defaultPrompt; });
  $("prompt-cancel").addEventListener("click", () => $("prompt-dialog").close());
  $("prompt-save").addEventListener("click", async () => {
    const value = $("prompt-text").value.trim();
    if (!value) { $("prompt-error").textContent = "Введите промпт или восстановите вариант по умолчанию."; return; }
    $("prompt-save").disabled = true;
    try {
      const data = await fs.getDataFolder();
      const file = await data.createFile(promptFile(), { overwrite: true });
      await file.write(value);
      activePrompt = value;
      $("prompt-dialog").close();
      log("Промпт сохранён. Он будет использован при следующей ретуши.");
    } catch (e) { $("prompt-error").textContent = "Не удалось сохранить: " + e.message; }
    finally { $("prompt-save").disabled = false; }
  });

  // ---------- статус сервера ----------
  let busy = false;
  async function checkServer() {
    if (busy) return; // во время обработки не дёргаем сервер лишний раз
    const el = $("server-status");
    try {
      const r = await fetch(SERVER + "/health");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      if (!j.capabilities || !["auto_quality", "native_resolution", "raw_input", "jsx_automation"].every(c => j.capabilities.includes(c))) {
        throw new Error("Запущена старая версия сервера. Перезапустите Chinminator из обновлённой папки проекта.");
      }
      await requirePrompt();
      if (j.engine && j.default_prompt && syncedEngine !== j.engine) {
        activeEngine = j.engine;
        modelLabel = j.model;
        defaultPrompt = j.default_prompt;
        const data = await fs.getDataFolder();
        const saved = (await data.getEntries()).find(entry => entry.name === promptFile());
        activePrompt = saved ? (await saved.read()).trim() : defaultPrompt;
        if (!activePrompt) activePrompt = defaultPrompt;
        syncedEngine = j.engine;
      }
      $("model-label").textContent = j.model || modelLabel;
      fixedSteps = j.fixed_steps || null;
      $("max-detail").disabled = Boolean(fixedSteps);
      if (fixedSteps) {
        $("quality-note").textContent = `Исходное разрешение области · фиксированное число шагов: ${fixedSteps}`;
      } else {
        $("quality-note").textContent = activeEngine === "flux3"
          ? "16 шагов · исходное разрешение; «Максимум деталей» — 1,5×"
          : "Исходное разрешение области · автошаги по её размеру";
      }
      const ok = j.qwen_backend && !j.qwen_backend.startsWith("UNAVAILABLE");
      el.textContent =
        (ok ? "● Сервер готов: " : "▲ Выбранная модель недоступна: ") +
        j.qwen_backend;
      el.className = "status " + (ok ? "ok" : "warn");
    } catch (e) {
      el.textContent = "✕ Нет связи с " + SERVER + ": " + (e.message || String(e));
      console.error("Chinminator health request failed", e);
      el.className = "status err";
    }
  }
  checkServer();
  setInterval(checkServer, HEALTH_MS);

  // ---------- слайдеры ----------
  $("threshold").addEventListener("input", (e) =>
    ($("v-thresh").textContent = e.target.value));


  const currentMode = () =>
    document.querySelector('input[name="mode"]:checked').value;

  function setBusy(on) {
    busy = on;
    $("btn-apply").textContent = on ? "Обработка…" : "Запустить ретушь";
    $("btn-apply").disabled = on;
    $("btn-analyze").disabled = on;
    $("btn-prompt").disabled = on;
    $("btn-record-action").disabled = on;
    $("btn-image-processor").disabled = on;
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
  async function inModal(task, commandName, context) {
    if (context) {
      if (context.isCancelled) throw new Error("Операция отменена");
      return await task();
    }
    return await core.executeAsModal(task, { commandName });
  }

  async function exportDoc(doc, context) {
    return await inModal(async () => {
      const pixels = await imaging.getPixels({ documentID: doc.id, components: 3, colorSpace: "RGB", componentSize: 8, applyAlpha: true });
      try {
        const data = await pixels.imageData.getData({ chunky: true });
        const envelope = new Uint8Array(16 + data.byteLength);
        envelope.set([67,72,73,78,82,65,87,49]);
        const header = new DataView(envelope.buffer);
        header.setUint32(8, pixels.imageData.width, true); header.setUint32(12, pixels.imageData.height, true);
        envelope.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 16);
        return envelope.buffer;
      } finally { pixels.imageData.dispose(); }
    }, "Read Chinminator pixels", context);
  }

  function currentSettings() {
    let quality = { auto_quality: 1, native_resolution: 1 };
    if (fixedSteps || $("max-detail").checked) {
      quality = { auto_quality: 0, steps: fixedSteps || 40, native_resolution: 1 };
    }
    return {
      version: 1, mode: currentMode(),
      params: {
        thresholds: { submental_ratio: parseFloat($("threshold").value) },
        prompt: activePrompt,
        quality,
      },
    };
  }

  function buildForm(bytes, mode, settings) {
    const fd = new FormData();
    fd.append("file", new Blob([bytes], { type: "application/octet-stream" }), "photo.chinraw");
    fd.append("mode", mode);
    fd.append("params", JSON.stringify((settings || currentSettings()).params));
    return fd;
  }

  $("btn-image-processor").addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    try {
      await requirePrompt();
      const response = await fetch(SERVER + "/automation/settings", {
        method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(currentSettings()),
      });
      if (!response.ok) throw new Error(await failText(response));
      log("Настройки сохранены. Запустите scripts/Install-ImageProcessor-Action.jsx через Файл → Сценарии → Обзор. В Обработчике изображений выберите Chinminator Image Processor → Retouch.");
    } catch (e) { log("Настройки сценария: " + e.message); }
    finally { setBusy(false); }
  });

  $("btn-record-action").addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    try {
      await requirePrompt();
      if (typeof ps.action.recordAction !== "function") throw new Error("Запись операций требует Photoshop 25 или новее");
      await ps.action.recordAction({ name: "Chinminator — ретушь", methodName: "chinminatorAction" },
        { chinminatorSettings: JSON.stringify(currentSettings()) });
      log("Шаг отправлен в запись операции. Остановите запись в панели «Операции». Ретушь выполнится при воспроизведении.");
    } catch (e) { log("Запись шага: " + e.message); }
    finally { setBusy(false); }
  });

  chinminatorPlayback = async function(context, info) {
    if (busy) throw new Error("Chinminator уже обрабатывает изображение");
    const settings = JSON.parse(info.chinminatorSettings);
    if (settings.version !== 1 || !["auto", "forced"].includes(settings.mode) ||
        !settings.params || typeof settings.params.prompt !== "string" ||
        !settings.params.quality || settings.params.quality.native_resolution !== 1) {
      throw new Error("Неподдерживаемые настройки шага Chinminator. Запишите шаг заново.");
    }
    setBusy(true);
    try {
      // A recorded step always targets the current document, never the panel batch/side-by-side settings.
      const properties = ["documentID", "title", "mode", "depth"];
      const values = await ps.action.batchPlay(properties.map(property => ({
        _obj: "get", _target: [{ _property: property }, { _ref: "document", _enum: "ordinal", _value: "targetEnum" }],
      })), {});
      for (const value of values) if (value._obj === "error") throw new Error(value.message);
      const doc = {
        id: values[0].documentID, title: values[1].title,
        mode: values[2].mode._value === "RGBColor" ? constants.DocumentMode.RGB : "unsupported",
        bitsPerChannel: values[3].depth === 8 ? constants.BitsPerChannelType.EIGHT : "unsupported",
        createPixelLayer: async () => {
          const result = await ps.action.batchPlay([{ _obj: "make", _target: [{ _ref: "layer" }], using: { _obj: "layer", name: "Chinminator Fixed" } }], {});
          if (result[0]._obj === "error") throw new Error(result[0].message);
          return { id: result[0].layerID };
        },
      };
      await processDoc(doc, settings.mode, false, settings, context);
      return info;
    } catch (e) {
      log("Ошибка операции: " + e.message);
      throw e; // Photoshop must stop the Action rather than silently save an unretouched image.
    } finally { setBusy(false); }
  };

  async function failText(r) {
    try { return (await r.json()).detail || r.statusText; }
    catch (_) { return r.statusText; }
  }

  // ---------- предпросмотр анализа (шаги 1–3, без Qwen-Image 2.1) ----------
  $("btn-analyze").addEventListener("click", async () => {
    setBusy(true);
    try {
      const doc = app.activeDocument;
      assertSupported(doc);
      await requirePrompt();
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
      } else {
        $("preview-img").src = "";
        $("preview-box").classList.add("hidden");
        $("metrics").textContent = JSON.stringify(j, null, 2);
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
  async function processDoc(doc, mode, sideBySide, settings, context) {
    assertSupported(doc);
    log(`«${doc.title}»: экспорт изображения…`);
    if (!settings) await requirePrompt();
    const bytes = await exportDoc(doc, context);
    log(`«${doc.title}»: ретушь ${modelLabel} на GPU, ожидайте…`);
    const fmt = sideBySide ? "png" : "raw";
    const r = await fetch(`${SERVER}/process?sidebyside=${sideBySide ? 1 : 0}&format=${fmt}`, {
      method: "POST", body: buildForm(bytes, mode, settings),
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
    if (context && context.isCancelled) throw new Error("Операция отменена");

    if (!sideBySide && (!Number.isInteger(width) || !Number.isInteger(height) ||
        width <= 0 || height <= 0 || body.byteLength !== width * height * 3)) {
      throw new Error("Сервер вернул неверные размеры RGB-изображения");
    }

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
    await inModal(async () => {
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
    }, "Apply Chinminator", context);
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
