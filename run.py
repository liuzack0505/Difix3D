from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image
from tqdm import tqdm

input_folder=Path("/app/data/orthogonal_input")
output_folder=Path("/app/data/orthogonal_output")

if not output_folder.exists():
    output_folder.mkdir(parents=True, exist_ok=True)

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

for input_path in tqdm(input_folder.glob("*.png"), desc="Processing images"):
    input_image = load_image(str(input_path))
    prompt = "remove degradation"

    output_image = pipe(prompt, image=input_image, num_inference_steps=2, timesteps=[199], guidance_scale=0.0).images[0]
    output_image = output_image.resize(input_image.size, Image.LANCZOS)
    output_image.save(str(output_folder / input_path.name))