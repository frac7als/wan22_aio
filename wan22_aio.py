import os
import subprocess
import modal

# =========================
# Fast-start ComfyUI (WAN 2.2 I2V safetensors, VHS only)
# =========================

# ------ Base image & system deps ------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    .pip_install(
        # Core runtime (Torch/CUDA comes from comfy --nvidia, cached on Modal)
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        # VHS dependencies
        "av",
        "decord",
        "einops",
    )
    .run_commands(
        # Install ComfyUI with NVIDIA stack (pulls CUDA torch/vision as needed)
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        # Enable HF transfer acceleration globally
        'bash -lc \'echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/profile\'',
    )
    # ------ Custom nodes ------
    .run_commands(
        # VideoHelperSuite (VHS nodes & Video_Upscale_With_Model)
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        # If VHS ships a requirements.txt, install it (don’t fail if missing)
        "bash -lc 'REQ=/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; "
        "if [ -f \"$REQ\" ]; then pip install -r \"$REQ\"; fi'",
    )
    # ------ Runtime env vars ------
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
        # Optional: pin Phr00t repo revision globally (can override at deploy time)
        # "WAN_AIO_REV": "main",
    })
)

# Shared HF cache volume so model blobs are reused
HF_VOL = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


def _ensure_dirs():
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        os.makedirs(d, exist_ok=True)


def _safe_ln(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(f"ln -sf '{src}' '{dst}'", shell=True, check=True)


def hf_warm_and_link():
    """
    One-time model warm (cached in /cache volume) with EXACT filenames your graph expects.
    Keeps Phr00t UNET; revision can be pinned via WAN_AIO_REV (env var).
    """
    from huggingface_hub import hf_hub_download

    _ensure_dirs()

    # Allow pinning the Phr00t repo revision (commit/tag). Default to "main".
    UNET_REV = os.environ.get("WAN_AIO_REV", "main")

    # --- UNET (WAN 2.2 I2V Rapid AIO, safetensors from Phr00t) ---
    unet = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
        revision=UNET_REV,
    )
    _safe_ln(
        unet,
        "/root/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-rapid-aio.safetensors",
    )

    # --- VAE (WAN 2.1) ---
    vae = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
        revision="main",
    )
    _safe_ln(vae, "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors")

    # --- Text encoder (UMT5) ---
    t5 = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
        revision="main",
    )
    _safe_ln(
        t5,
        "/root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    )

    # --- Upscaler (Remacri) ---
    try:
        remacri = hf_hub_download(
            repo_id="fofr/comfyui",
            filename="upscale_models/4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
            revision="main",
        )
    except Exception:
        remacri = hf_hub_download(
            repo_id="FacehugmanIII/4x_foolhardy_Remacri",
            filename="4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
            revision="main",
        )
    _safe_ln(
        remacri,
        "/root/comfy/ComfyUI/models/upscale_models/4x_foolhardy_Remacri.pth",
    )

    # Optional: list to verify
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        subprocess.run(f"ls -lh {d}", shell=True, check=False)


# Bake the warm step into the image (cached) so first UI boot is quick
image = image.run_function(hf_warm_and_link, volumes={"/cache": HF_VOL})

app = modal.App("wan22-i2v-rapid-aio-fast", image=image)


@app.function(
    gpu="L40S",
    max_containers=1,
    timeout=60 * 60,
    volumes={"/cache": HF_VOL},
)
@modal.web_server(8000, startup_timeout=600)
def ui():
    # Runtime fallbacks for env vars
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
    )
    os.environ.setdefault("WAN_AIO_REV", "main")

    # Tiny CUDA warmup so first op feels instant
    subprocess.run(
        "python - <<'PY'\n"
        "import torch\n"
        "torch.cuda.init()\n"
        "_ = torch.randn(1, device='cuda')\n"
        "print('CUDA warmup done')\n"
        "PY",
        shell=True,
        check=False,
    )

    # Launch ComfyUI
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
