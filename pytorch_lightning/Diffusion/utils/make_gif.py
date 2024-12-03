import os
import yaml
from PIL import Image

# YAML 파일 로드
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# 파일명에서 숫자를 추출하는 함수
def extract_number(file_name):
    try:
        return int(os.path.splitext(file_name)[0])  # 확장자를 제거한 후 숫자로 변환
    except ValueError:
        return float("inf")  # 숫자가 아니면 뒤로 정렬

# GIF 생성 함수
def create_gif_from_images(config):
    """
    Config 파일에서 디렉토리와 설정을 읽어 GIF 생성

    Args:
        config (dict): 설정 값 포함
    """
    directory = config["gif_config"]["image_directory"]
    output_file = config["gif_config"]["output_gif"]
    duration = config["gif_config"].get("duration", 100)

    images = []
    for file_name in sorted(os.listdir(directory), key=extract_number):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, file_name)
            img = Image.open(file_path)
            images.append(img)


    if not images:
        print("이미지 파일이 없습니다.")
        return

    # GIF 생성 및 저장
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF가 성공적으로 저장되었습니다: {output_file}")

# 실행
if __name__ == "__main__":
    config_path = "/home/work/reality/hojae/genAI/pytorch_lightning/Diffusion/config/gif/gif_config.yaml"
    config = load_config(config_path)
    create_gif_from_images(config)