"""Pinned official InternVL/DeepSeek wrappers for the paper's external comparison."""

import tempfile
import time

from src.recognition.internvl_preprocess import load_pixels


class ExternalModel:
    def __init__(self, spec):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.spec = torch, spec
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["directory"],
            trust_remote_code=True,
            local_files_only=True,
            use_fast=spec["role"] != "internvl",
        )
        options = (
            {"use_flash_attn": spec["settings"]["use_flash_attn"]}
            if spec["role"] == "internvl"
            else {"_attn_implementation": "flash_attention_2"}
        )
        self.model = (
            AutoModel.from_pretrained(
                spec["directory"],
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                **options,
            )
            .eval()
            .to("cuda:0")
        )
        if spec["role"] == "deepseek":
            self.model.generation_config.do_sample = False
            self.model.generation_config.num_beams = 1

    def infer(self, image):
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        settings = self.spec["settings"]
        with self.torch.inference_mode():
            if self.spec["role"] == "internvl":
                pixels = load_pixels(image, settings["max_tiles"], settings["tile_size"])
                pixels = pixels.to(device="cuda:0", dtype=self.torch.bfloat16)
                text = self.model.chat(
                    self.tokenizer,
                    pixels,
                    self.spec["prompt"],
                    {key: settings[key] for key in ("max_new_tokens", "do_sample", "num_beams")},
                    history=None,
                    return_history=False,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="deepseek-ocr-") as temporary:
                    text = self.model.infer(
                        self.tokenizer,
                        prompt=self.spec["prompt"],
                        image_file=str(image),
                        output_path=temporary,
                        base_size=settings["base_size"],
                        image_size=settings["image_size"],
                        crop_mode=settings["crop_mode"],
                        eval_mode=True,
                        save_results=False,
                    )
        if not isinstance(text, str):
            raise ValueError("Native model wrapper did not return text")
        self.torch.cuda.synchronize()
        return {
            "text": text,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "parser": "native model wrapper; no extra cleanup",
            "hit_token_limit": None,
            "token_limit_note": "Native text-only interface does not expose generated token IDs",
        }
