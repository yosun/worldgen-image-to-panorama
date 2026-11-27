# WorldGen image-to-panorama on Replicate

This Cog wrapper calls the image-to-panorama stage of WorldGen at upstream revision `7ce7b2767fdf31e2727b69a2e61e2e950e3a017f`. It uses DA-2, FLUX.1-Fill-dev, WorldGen's image LoRA and the low-VRAM Nunchaku transformer. It returns the panorama as PNG. It does not produce a 3D scene.

The first request on each worker downloads pinned model files, including roughly 17 GB of checkpoint data, and initializes the models. It requires network access and may take a long time. Worker storage is temporary. Subsequent requests can reuse loaded models while the worker remains online.

The private model is published at `yosun/worldgen-image-to-panorama` as version `bf712f97a9a290ade99b5b190d3bbdb108cd5af816ddba4ae0e86ce0b1306de5`. The deployment with the same name references that version, uses an L40S and scales between zero and one instance. The model's default hardware is an A100.

An L40S test using WorldGen's street sample succeeded on November 27, 2026. It produced a 2048 × 1024 RGB PNG. Model download and initialization took 134.68 seconds, generation took 28.37 seconds, and peak allocated PyTorch GPU memory was 12,393,016,832 bytes. Replicate reported 163.77 seconds of prediction time and 238.03 seconds total, including startup. These are measurements from one cold run. The output was verified as a PNG.

The Cog SDK and Coglet are pinned to 0.23.1 to avoid the earlier `BaseRunner` startup error. Image projection uses batches of 256 queries instead of the upstream default of 8192 to reduce peak memory. Batching preserves the nearest-neighbor algorithm, although floating-point tie behavior may vary. The earlier 8192-query version exhausted L40S memory on the test image.

To start a prediction against the deployment, make the Replicate API token and gated Hugging Face token available as `REPLICATE_API_TOKEN` and `HF_TOKEN`, then run:

```python
import json
import os
from urllib.request import Request, urlopen

payload = {
    "input": {
        "image": "https://example.com/your-image.png",
        "prompt": "A quiet street in daylight",
        "hf_token": os.environ["HF_TOKEN"],
    }
}
request = Request(
    "https://api.replicate.com/v1/deployments/yosun/worldgen-image-to-panorama/predictions",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['REPLICATE_API_TOKEN']}",
        "Content-Type": "application/json",
        "User-Agent": "worldgen-client/1.0",
    },
    method="POST",
)
with urlopen(request) as response:
    prediction = json.load(response)
print(prediction["id"], prediction["status"])
```

Poll `GET https://api.replicate.com/v1/predictions/{id}` with the Replicate authorization header to check status and retrieve the output URL. The successful test returned the URL as the first item in `output`. The first request on an idle worker may take substantially longer while weights download.

Each prediction requires a Hugging Face token in the secret input `hf_token`. Its account must have access to [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), whose license governs use of those model weights. The token is passed to Hugging Face only for the gated model download. Never put tokens into source files or command history.

Build and push with a current Cog release and an AMD64 Docker build environment. Select compatible NVIDIA GPU hardware on Replicate, and submit an image plus the secret token. Additional input images and downstream reconstruction have not been validated.

WorldGen source and its released LoRA are licensed under Apache 2.0. The FLUX.1-Fill-dev base model and the quantized FLUX Fill transformer have separate non-commercial licenses with restrictions on end-user-facing use. Review the applicable model terms before deployment.
