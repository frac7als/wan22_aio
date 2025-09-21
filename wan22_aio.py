import subprocess
import os
import modal

# Optimized image with only necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "wget",
    )
    # Essential Python dependencies for the workflow
    .pip_install(
        "gguf",
        "llama-cpp-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "moviepy",
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.5.1",
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59"
    )
)

# Install necessary custom nodes based on workflow analysis
image = image.run_commands(
    # VideoHelperSuite - Required for VHS_VideoCombine nodes
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    # Essential utilities
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
)

def hf_download():
    from huggingface_hub import hf_hub_download
    
    # Download VAE model - Required by VAELoader node
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    # Download text encoder - Required by CLIPLoader node
    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -sf {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

    # Download the normal diffusion model - Required by UNETLoader node
    diffusion_model = hf_hub_download(
        repo_id="Phr00t/WAN2.2-14B-Rapid-AllInOne",
        filename="wan2.2-i2v-rapid-aio.safetensors",
        cache_dir="/cache",
    )
    
    # Ensure diffusion models directory exists
    diffusion_dir = "/root/comfy/ComfyUI/models/diffusion_models"
    os.makedirs(diffusion_dir, exist_ok=True)
    
    subprocess.run(
        f"ln -sf {diffusion_model} {os.path.join(diffusion_dir, 'wan2.2-i2v-rapid-aio.safetensors')}",
        shell=True,
        check=True,
    )

    # Download upscaling model - Required by Video_Upscale_With_Model node
    upscale_model = hf_hub_download(
        repo_id="FacehugmanIII/4x_foolhardy_Remacri",
        filename="4x_foolhardy_Remacri.pth",
        cache_dir="/cache",
    )
    
    # Ensure upscale models directory exists
    upscale_dir = "/root/comfy/ComfyUI/models/upscale_models"
    os.makedirs(upscale_dir, exist_ok=True)
    
    subprocess.run(
        f"ln -sf {upscale_model} {os.path.join(upscale_dir, '4x_foolhardy_Remacri.pth')}",
        shell=True,
        check=True,
    )

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # Install huggingface_hub with hf_transfer support
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

app = modal.App(name="wan-video-generation", image=image)

@app.function(
    max_containers=1,
    gpu="H100",  # Upgraded GPU for full-precision model
    volumes={"/cache": vol},
    timeout=3600,  # 1 hour timeout for video generation
)
@modal.concurrent(max_inputs=3)  # Reduced concurrency for full model
@modal.web_server(8000, startup_timeout=120)  # Increased startup timeout
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

# Optional: CLI function for batch processing
@app.function(
    gpu="H100",
    volumes={"/cache": vol},
    timeout=3600,
)
def generate_video(
    image_path: str,
    positive_prompt: str,
    negative_prompt: str = "Blurry face, distorted features, extra fingers, poorly drawn hands",
    width: int = 720,
    height: int = 720,
    length: int = 125,
    steps: int = 7,
    cfg: float = 1.0,
    seed: int = -1
):
    """
    Generate video from image using the Wan model
    
    Args:
        image_path: Path to input image
        positive_prompt: Description of desired video
        negative_prompt: Things to avoid in the video
        width: Output width (default 720)
        height: Output height (default 720)
        length: Number of frames (default 125)
        steps: Sampling steps (default 7)
        cfg: CFG scale (default 1.0)
        seed: Random seed (-1 for random)
    """
    import json
    import random
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    # Create workflow JSON programmatically
    workflow = {
        "prompt": {
            # Add nodes based on the analyzed workflow
            # This would be a programmatic version of the workflow
        }
    }
    
    # Execute the workflow
    # Implementation would depend on ComfyUI's API
    pass

if __name__ == "__main__":
    # For local development/testing
    pass
