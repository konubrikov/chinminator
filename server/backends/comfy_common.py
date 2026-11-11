"""HTTP transport for the local ComfyUI API."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def comfy_is_up(url: str = COMFY_URL) -> bool:
    try:
        with urllib.request.urlopen(url + "/system_stats", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def post_prompt(graph: dict) -> str:
    """Отправляет workflow, возвращает prompt_id."""
    cid = uuid.uuid4().hex
    req = urllib.request.Request(
        COMFY_URL + "/prompt",
        data=json.dumps({"prompt": graph, "client_id": cid}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected workflow (HTTP {e.code}): {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"ComfyUI connection failed: {e}") from e


def wait_done(prompt_id: str, timeout_s: int = 1800) -> dict:
    """Ждёт завершения промпта, возвращает history entry outputs."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with urllib.request.urlopen(
                f"{COMFY_URL}/history/{prompt_id}", timeout=10) as r:
            hist = json.loads(r.read())
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI execution failed: " + json.dumps(status.get("messages", []), ensure_ascii=False))
            return entry.get("outputs", {})
        time.sleep(0.5)
    raise RuntimeError(f"ComfyUI: таймаут ожидания {prompt_id}")


def fetch_image(outputs: dict) -> np.ndarray:
    """Достаёт первое SaveImage/VAEDecode-изображение как BGR uint8."""
    for node_out in outputs.values():
        if "images" in node_out:
            im = node_out["images"][0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im["type"]})
            with urllib.request.urlopen(f"{COMFY_URL}/view?{q}", timeout=60) as r:
                pil = Image.open(BytesIO(r.read())).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    raise RuntimeError("В ответе ComfyUI нет изображения")


def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_node_field(graph: dict, class_type: str, field: str, value) -> None:
    """Меняет значение поля у первого узла данного класса."""
    for node in graph.values():
        if node.get("class_type") == class_type:
            node["inputs"][field] = value
            return
    raise KeyError(f"Узел {class_type} не найден в workflow")
