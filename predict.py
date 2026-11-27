"""Serve WorldGen's image-to-panorama stage using its low-VRAM pipeline."""

import json
import tempfile
import time
from functools import partial

import torch
from cog import BasePredictor, Input, Path, Secret
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageOps

from da2.model.spherevit import SphereViT
from nunchaku import NunchakuFluxTransformer2dModel
from worldgen import WorldGen
import worldgen.worldgen as worldgen_module
from worldgen.models.flux_pano_fill_pipeline import FluxFillPipeline
from worldgen.pano_depth import DA2_CONFIG
from worldgen.utils.lora_utils import load_and_fix_lora


class Predictor(BasePredictor):
    def setup(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("WorldGen requires a CUDA GPU with bfloat16 support")
        # Bound the nearest-neighbor matrix used to map an image onto the panorama.
        worldgen_module.map_image_to_pano = partial(worldgen_module.map_image_to_pano, nn_batch=256)
        self.worldgen = None

    def _load_model(self, token):
        started = time.monotonic()
        print("Downloading pinned WorldGen models; the first prediction may take time.", flush=True)
        fill_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            revision="7326a96224b2513681b5d97281c83af967926bb6",
            token=token,
            allow_patterns=[
                "model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                "tokenizer/*", "tokenizer_2/*", "vae/*",
            ],
        )
        quantized_path = snapshot_download(
            repo_id="mit-han-lab/svdq-int4-flux.1-fill-dev",
            revision="e331143be73c40573ae227a4fac14105a3dfd6c7",
            allow_patterns=["config.json", "*.safetensors"],
        )
        depth_path = snapshot_download(
            repo_id="haodongli/DA-2",
            revision="0d55ccb5e46b8ed4715fae3a4c04fc897f1689f3",
            allow_patterns=["model.safetensors"],
        )
        lora_path = hf_hub_download(
            repo_id="LeoXie/WorldGen",
            revision="2e2b2cb4dc3144a65481f0ca26b892584436868c",
            filename="models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
        )

        # WorldGen's constructor loads by unpinned repository names. Assemble the
        # same image-mode components from pinned snapshots instead.
        worldgen = WorldGen.__new__(WorldGen)
        worldgen.device = torch.device("cuda")
        worldgen.mode = "i2s"
        worldgen.resolution = 1600
        worldgen.depth_model = SphereViT.from_pretrained(depth_path, config=DA2_CONFIG)
        worldgen.depth_model.eval()
        worldgen.depth_model = worldgen.depth_model.to(worldgen.device)

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(quantized_path, offload=True)
        worldgen.pano_gen_model = FluxFillPipeline.from_pretrained(
            fill_path, transformer=transformer, torch_dtype=torch.bfloat16, device=worldgen.device
        )
        lora, _ = load_and_fix_lora(lora_path)
        transformer.update_lora_params(lora)
        worldgen.pano_gen_model.enable_model_cpu_offload()
        worldgen.pano_gen_model.enable_vae_tiling()
        self.worldgen = worldgen
        print(json.dumps({"event": "model_loaded", "seconds": round(time.monotonic() - started, 2)}), flush=True)

    def predict(
        self,
        image: Path = Input(description="Source photograph"),
        hf_token: Secret = Input(description="Hugging Face token with access to FLUX.1-Fill-dev"),
        prompt: str = Input(description="Optional scene description", default=""),
    ) -> Path:
        if self.worldgen is None:
            self._load_model(hf_token.get_secret_value())

        with Image.open(str(image)) as source:
            source_image = ImageOps.exif_transpose(source).convert("RGB")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            panorama = self.worldgen.generate_pano(prompt=prompt, image=source_image)
        output = Path(tempfile.mkdtemp()) / "panorama.png"
        panorama.save(str(output), format="PNG")
        torch.cuda.synchronize()
        print(json.dumps({
            "event": "prediction_complete",
            "seconds": round(time.monotonic() - started, 2),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "size": list(panorama.size),
        }), flush=True)
        return output
