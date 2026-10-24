from __future__ import annotations

import base64
import importlib.metadata
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


class CachedVLM:
    def infer(self, sample: dict) -> dict:
        if "vlm" not in sample:
            raise ValueError(f"Missing VLM observation for {sample['id']}")
        return dict(sample["vlm"])


class PaddleOCRAdapter:
    """PaddleOCR 3.2.0 contract; only explicit local model directories, CPU by default."""

    def __init__(self, detection_dir: Path, recognition_dir: Path, device: str = "cpu"):
        if not detection_dir.is_dir() or not recognition_dir.is_dir():
            raise ValueError("Both reviewed local Paddle model directories are required")
        if device != "cpu":
            raise ValueError("Stage 1 adapter permits CPU only; review GPU allocation first")
        if importlib.metadata.version("paddleocr") != "3.2.0":
            raise RuntimeError("This adapter targets paddleocr==3.2.0; revalidate other versions")
        from paddleocr import PaddleOCR

        self.model = PaddleOCR(
            text_detection_model_dir=str(detection_dir),
            text_recognition_model_dir=str(recognition_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def infer(self, image: Path) -> dict:
        start = time.perf_counter()
        pages = list(self.model.predict(input=str(image)))
        texts, scores, polygons = [], [], []
        for page in pages:
            texts.extend(page["rec_texts"])
            scores.extend(float(x) for x in page["rec_scores"])
            polygons.extend(p.tolist() if hasattr(p, "tolist") else p for p in page["rec_polys"])
        return {
            "text": "\n".join(texts),
            "confidences": scores,
            "polygons": polygons,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


class QwenVLHTTPAdapter:
    """OpenAI-compatible local Qwen-VL serving API; no SDK, weights or remote code loading."""

    def __init__(self, endpoint: str, model: str, timeout: float = 120):
        url = urllib.parse.urlparse(endpoint)
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
        ):
            raise ValueError("Only explicit loopback Qwen endpoints are allowed in Stage 1")
        if not model or timeout <= 0:
            raise ValueError("Model ID and positive timeout are required")
        self.endpoint, self.model, self.timeout = endpoint, model, timeout

    def infer(self, image: Path) -> dict:
        mime = mimetypes.guess_type(image.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("Use PNG/JPEG/WebP images")
        if image.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Image exceeds Stage 1 20 MiB request limit")
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe visible text only, in reading order. "
                            "Do not guess invisible text. Return empty text if no text is visible.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get("QWEN_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["QWEN_API_KEY"]
        request = urllib.request.Request(
            self.endpoint, json.dumps(payload).encode(), headers, method="POST"
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise ValueError("Redirects are disabled for image privacy")

        start = time.perf_counter()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
        with opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("VLM response content is not a string")
        return {
            "text": text,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "usage": result.get("usage"),
        }
