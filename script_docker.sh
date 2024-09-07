# use Docker
docker build -f Dockerfile -t storydiffusion:dev .
docker run -it --name storydiffusion  --runtime=nvidia --gpus all storydiffusion:dev
cd /workspace/StoryDiffusion; 
python gradio_app_sdxl_specific_id_low_vram.py