def hf_warm_and_link():
    """
    Download required models to /cache and symlink them into ComfyUI model dirs
    with EXACT filenames your workflow expects.
    """
    import os
    import subprocess
    from huggingface_hub import hf_hub_download

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

    _ensure_dirs()

    # --- UNET (WAN 2.2 I2V Rapid AIO, safetensors) ---
    unet = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(unet, "/root/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-rapid-aio.safetensors")

    # --- VAE (WAN 2.1) ---
    vae = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(vae, "/root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors")

    # --- Text encoder (UMT5) ---
    t5 = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    _safe_ln(t5, "/root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

    # --- Upscaler (VideoHelperSuite) ---
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

    # Optional: quick listings for sanity
    for d in [
        "/root/comfy/ComfyUI/models/diffusion_models",
        "/root/comfy/ComfyUI/models/vae",
        "/root/comfy/ComfyUI/models/text_encoders",
        "/root/comfy/ComfyUI/models/upscale_models",
    ]:
        subprocess.run(f"ls -lh {d}", shell=True, check=False)
