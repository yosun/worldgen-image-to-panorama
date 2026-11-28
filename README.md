# WorldGen image-to-panorama on Replicate

This Cog wrapper calls the image-to-panorama stage of WorldGen at upstream revision `7ce7b2767fdf31e2727b69a2e61e2e950e3a017f`. It uses FLUX.1-Fill-dev, WorldGen's image LoRA and the low-VRAM Nunchaku transformer. It returns the panorama as PNG. It does not produce a 3D scene.

The first request on each worker downloads pinned model files, including roughly 17 GB of checkpoint data, and initializes the models. It requires network access and may take a long time. Worker storage is temporary. Subsequent requests can reuse loaded models while the worker remains online.

The public community model is published at `yosun/worldgen-image-to-panorama`. There is also a separate L40S deployment, configured to scale between zero and one instance. Requests sent to the deployment are billed for instance setup, processing, and idle time until scale-down; the instance stops incurring charges once it reaches zero. Requests sent directly to the public model use shared-pool billing, where Replicate charges for prediction processing but not instance setup or idle time. Model downloads and initialization performed inside this predictor are part of prediction processing. See [Replicate billing](https://replicate.com/docs/topics/billing).

Ordinary photographs are projected into the center of the panorama as perspective images, assuming a 90-degree horizontal field of view and square pixels. The photo is resampled into equirectangular projection, so it will look distorted in the flat panorama; viewing the center through a perspective panorama viewer reconstructs the original framing. The surrounding area is generated. The field of view is an assumption and may not match every camera.

The pipeline redraws an overlap around the source-photo boundary, runs an additional inpainting pass over that join, and blends a narrow band at the 360-degree wrap. The middle of the source photo is preserved; its outer edges can change. These steps reduce visible joins but cannot guarantee coherent geometry or eliminate artifacts.

A public-model test on WorldGen's street photo produced a 2048 × 1024 panorama whose left and right edge pixels matched. The reprojected source view had a mean absolute RGB difference of 11/255 at 512-pixel width. Road markings still showed some mismatch. The two inpainting passes took 57.84 seconds, and the entire prediction took 86.10 seconds including model loading, with peak allocated PyTorch GPU memory of 11,007,391,232 bytes. Other photos have not been systematically evaluated; times vary with hardware and worker state.

The Cog SDK and Coglet are pinned to 0.23.1 to avoid the earlier `BaseRunner` startup error. The original upstream spherical-ray mapping discarded the test photo completely; this wrapper instead constructs perspective rays. Depth estimation is unnecessary for this image-to-panorama path because the mapper only consumes RGB and ray directions.

To call the public community model directly, set `REPLICATE_API_TOKEN` and `HF_TOKEN`, and set `WORLDGEN_MODEL_VERSION` to `db69b14c61166130a4090d9bf26a18f6e85440a375f95aec6d20c1a36f30cd29` or another desired ID from the model's Versions page. Community model calls must specify a version:

```python
import json
import os
from urllib.request import Request, urlopen

payload = {
    "version": os.environ["WORLDGEN_MODEL_VERSION"],
    "input": {
        "image": "https://example.com/your-image.png",
        "prompt": "A quiet street in daylight",
        "hf_token": os.environ["HF_TOKEN"],
    }
}
request = Request(
    "https://api.replicate.com/v1/predictions",
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

Each prediction requires a Hugging Face token in the secret input `hf_token`. Its account must have access to [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), whose license governs use of those model weights. The token is sent to Hugging Face for gated access checks and model downloads. Never put tokens into source files or command history.

The current version checks gated-model access on every prediction, including when a worker has already loaded the model. Older published versions do not enforce this check on warm workers and remain callable while available.

Build and push with a current Cog release and an AMD64 Docker build environment. Select compatible NVIDIA GPU hardware on Replicate, and submit an image plus the secret token. Additional input images and downstream reconstruction have not been validated.
