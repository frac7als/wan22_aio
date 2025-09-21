import os
import subprocess
import modal

# =========================
# Minimal image for WAN 2.2 I2V (safetensors)
# =========================
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
        # Explicit CUDA 12.1 build of PyTorch + torchvision
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Core runtime
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        # HF with transfer acceleration
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
    )
    # Install ComfyUI
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        # QoL: enable HF transfer by default
        'bash -lc \'echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/profile\'',
    )
    # --- Required custom node(s) ONLY ---
    .run_commands(
        # VideoHelperSuite (VHS nodes & Video_Upscale_With_Model)
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        # Install VHS requirements if present
        "bash -lc 'REQ=/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; "
        "if [ -f \"$REQ\" ]; then pip install -r \"$REQ\"; fi'"
    )
)

# Shared cache volume for HF artifacts
HF_VOL = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

def _ensure_dirs():
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/loras",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        os.makedirs(d, exist_ok=True)

def _safe_ln(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(f"ln -sf '{src}' '{dst}'", shell=True, check=True)

def hf_warm_and_link():
    """
    Download required models to /cache and symlink them into ComfyUI model dirs
    with EXACT filenames your workflow expects.
    """
    from huggingface_hub import hf_hub_download

    _ensure_dirs()

    # --- UNET (WAN 2.2 I2V Rapid AIO, safetensors) ---
    unet = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(
        unet,
        "/root/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-rapid-aio.safetensors",
    )

    # --- VAE (WAN 2.1) ---
    vae = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir=_
