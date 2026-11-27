# WorldGen image-to-panorama on Replicate

This Cog wrapper calls the image-to-panorama stage of WorldGen at upstream revision `7ce7b2767fdf31e2727b69a2e61e2e950e3a017f`. It uses FLUX.1-Fill-dev, WorldGen's image LoRA and the low-VRAM Nunchaku transformer. It returns the panorama as PNG. It does not produce a 3D scene.

The first request on each worker downloads pinned model files, including roughly 17 GB of checkpoint data, and initializes the models. It requires network access and may take a long time. Worker storage is temporary. Subsequent requests can reuse loaded models while the worker remains online.

The private model and deployment are published at `yosun/worldgen-image-to-panorama`. The deployment uses an L40S and scales between zero and one instance. The model's default hardware is an A100.

Ordinary photographs are projected into the center of the panorama as perspective images, assuming a 90-degree horizontal field of view and square pixels. The photo is resampled into equirectangular projection, so it will look distorted in the flat panorama; viewing the center through a perspective panorama viewer reconstructs the original framing. The surrounding area is generated. The field of view is an assumption and may not match every camera.

A corrected L40S test on WorldGen's street photo produced a 2048 × 1024 panorama. The reprojected source view had a mean absolute RGB difference of 5.77/255 at 512-pixel width; generated areas included visible seams, and other photos have not been systematically evaluated. This cold run took 96.36 seconds to load the model and 25.21 seconds to generate, with peak allocated PyTorch GPU memory of 11,007,391,232 bytes. Times vary with worker startup and network conditions.

The Cog SDK and Coglet are pinned to 0.23.1 to avoid the earlier `BaseRunner` startup error. The original upstream spherical-ray mapping discarded the test photo completely; this wrapper instead constructs perspective rays. Depth estimation is unnecessary for this image-to-panorama path because the mapper only consumes RGB and ray directions.

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
