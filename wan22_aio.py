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
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
    )
    # Install ComfyUI
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        'bash -lc \'echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/profile\'',
    )
    # --- Required custom node(s) ---
    .run_commands(
        # Clone VHS
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        # Force-install VHS dependencies (ensures Video_Upscale_With_Model works)
        "pip install av decord einops opencv-python-headless",
    )
)

# Shared cache volume
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
    from huggingface_hub import hf_hub_download

    _ensure_dirs()

    # --- UNET (WAN 2.2 I2V Rapid AIO, safetensors) ---
    unet = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(unet, "/root/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-rapid-aio.safetensors")

    # --- VAE ---
    vae = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(vae, "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors")

    # --- Text encoder ---
    t5 = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(t5, "/root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

    # --- Upscaler (Remacri) ---
    try:
        remacri = hf_hub_download(
            repo_id="fofr/comfyui",
            filename="upscale_models/4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
        )
    except Exception:
        remacri = hf_hub_download(
            repo_id="FacehugmanIII/4x_foolhardy_Remacri",
            filename="4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
        )
    _safe_ln(remacri, "/root/comfy/ComfyUI/models/upscale_models/4x_foolhardy_Remacri.pth")

    # Optional sanity check
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        subprocess.run(f"ls -lh {d}", shell=True, check=False)

# Warm cache
image = image.run_function(hf_warm_and_link, volumes={"/cache": HF_VOL})

app = modal.App("wan22-i2v-rapid-aio", image=image)

@app.function(
    gpu="L40S",
    max_containers=1,
    timeout=60 * 60,
    volumes={"/cache": HF_VOL},
)
@modal.web_server(8000, startup_timeout=300)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
