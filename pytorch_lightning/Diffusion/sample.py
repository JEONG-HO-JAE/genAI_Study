import os
import yaml
import math
from PIL import Image
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def save_generated_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_tensor in enumerate(images):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()  # 텐서를 이미지로 변환
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')  # 정규화 후 uint8로 변환
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(output_dir, f"generated_{idx}.png"))
        
def sample(sampler_config):
    trainer_config = OmegaConf.load(os.path.join(sampler_config["model_dir"],'.hydra', 'config.yaml'))
    
    batch_size = sampler_config["batch_size"]
    sampling_timesteps = sampler_config["sampling_timesteps"]
    T = trainer_config.noise_steps
    
    ckpt = [f for f in os.listdir(sampler_config["model_dir"]) if f.endswith('.ckpt')]
    if not ckpt:
        raise FileNotFoundError("No .ckpt files found in the specified directory.")
    
    ckpt = os.path.join(sampler_config["model_dir"], ckpt[0])
    
    denoiser_module = hydra.utils.instantiate(trainer_config.denoiser_module)
    model_class = hydra.utils.get_class(trainer_config.model._target_)  # 모델 클래스 확인
    scheduler = hydra.utils.instantiate(trainer_config.scheduler)
    opt = hydra.utils.instantiate(trainer_config.optimizer)
    model = model_class.load_from_checkpoint(
        checkpoint_path=ckpt,
        denoiser_module=denoiser_module,
        opt=opt,
        variance_scheduler=scheduler
    )
    
    generated_images = model.sample(batch_size=batch_size, T=T, sampling_timesteps=sampling_timesteps)
    output_dir = os.path.join(sampler_config["model_dir"], "generated_images")
    save_generated_images(generated_images, output_dir)
    
    # # 정사각형 grid 크기 계산
    # grid_size = int(math.ceil(math.sqrt(batch_size)))
    # fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         idx = i * grid_size + j  # 이미지 인덱스
    #         if idx < batch_size:  # 배치 크기 초과 방지
    #             axs[i, j].imshow(generated_images[idx].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    #             axs[i, j].axis("off")
    #         else:
    #             axs[i, j].axis("off")  # 초과한 subplot은 비활성화

    # plt.tight_layout()
    # plt.show()
    
    
# 실행
if __name__ == "__main__":
    sampler_config_path = "/home/work/reality/hojae/genAI/pytorch_lightning/Diffusion/config/sampler/sample.yaml"
    sampler_config = load_config(sampler_config_path)
    sample(sampler_config)