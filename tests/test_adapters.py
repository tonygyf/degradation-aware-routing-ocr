import json
from unittest.mock import patch

from src.recognition.backends import PaddleOCRAdapter, QwenVLHTTPAdapter


def test_paddle3_output_contract(tmp_path):
    adapter = object.__new__(PaddleOCRAdapter)

    class Model:
        def predict(self, input):
            return [
                {
                    "rec_texts": ["你好", "OCR"],
                    "rec_scores": [0.8, 0.9],
                    "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                }
            ]

    adapter.model = Model()
    out = adapter.infer(tmp_path / "image.png")
    assert out["text"] == "你好\nOCR"
    assert out["confidences"] == [0.8, 0.9]
    assert out["latency_ms"] >= 0


def test_qwen_request_contract(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"mock-image")
    adapter = QwenVLHTTPAdapter("http://127.0.0.1:8000/v1/chat/completions", "Qwen/local")
    import io

    response = io.BytesIO(
        json.dumps(
            {"choices": [{"message": {"content": "测试"}}], "usage": {"total_tokens": 5}}
        ).encode()
    )
    with patch("urllib.request.OpenerDirector.open", return_value=response) as request:
        result = adapter.infer(image)
        payload = json.loads(request.call_args.args[0].data)
    assert payload["model"] == "Qwen/local"
    assert payload["temperature"] == 0
    assert result["text"] == "测试"
    assert result["usage"]["total_tokens"] == 5
