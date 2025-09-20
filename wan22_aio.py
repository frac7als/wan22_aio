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
        # Core runtime
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
        # Hugging Face with fast transfer
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
    )
    # Install ComfyUI (modern version to avoid node incompatibility)
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59"
    )
    # --- Required custom nodes ONLY ---
    .run_commands(
        # VideoHelperSuite for VHS nodes & Video_Upscale_With_Model
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        # Resolve any node Python deps
        "comfy node install --fast-deps --all"
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
    # Source: Phr00t/WAN2.2-14B-Rapid-AllInOne
    unet = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
        local_dir="/cache",
        local_dir_use_symlinks=False,
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
        local_dir="/cache",
        local_dir_use_symlinks=False,
    )
    _safe_ln(vae, "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors")

    # --- Text encoder (UMT5) ---
    t5 = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
        local_dir="/cache",
        local_dir_use_symlinks=False,
    )
    _safe_ln(
        t5,
        "/root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    )

    # --- Upscaler (VideoHelperSuite) ---
    # Try common mirrors; link to exact name your node expects
    remacri = None
    try:
        remacri = hf_hub_download(
            repo_id="fofr/comfyui",
            filename="upscale_models/4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
            local_dir="/cache",
            local_dir_use_symlinks=False,
        )
    except Exception:
        # Fallback mirror
        remacri = hf_hub_download(
            repo_id="FacehugmanIII/4x_foolhardy_Remacri",
            filename="4x_foolhardy_Remacri.pth",
            cache_dir="/cache",
            local_dir="/cache",
            local_dir_use_symlinks=False,
        )
    _safe_ln(remacri, "/root/comfy/ComfyUI/models/upscale_models/4x_foolhardy_Remacri.pth")

    # Optional: quick listings for sanity
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        subprocess.run(f"ls -lh {d}", shell=True, check=False)

# Warm cache in-image so first boot is ready
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
    # Launch ComfyUI at port 8000
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
