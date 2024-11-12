import torch
import torchvision.utils as vutils
import os


def reconstruct_test_images(config, model, dataloader, device):
    model.eval()
    with torch.no_grad():
        # 데이터 배치에서 이미지 추출
        example_images, _ = next(iter(dataloader))
        example_images = example_images.to(device)
        
        # 모델을 통해 예측
        results = model.forward(example_images)
        recons_images = results[0]
        
        # 원본 이미지와 재구성 이미지를 왼쪽-오른쪽으로 결합
        comparison = torch.cat([example_images, recons_images], dim=3)

        # 저장 경로 설정
        save_dir = os.path.join(config['logging_params']['save_dir'], 
                                config['logging_params']['name'],
                                "version_" + str(config['logging_params']['version']),
                                "Reconstruct_TestImages")
        os.makedirs(save_dir, exist_ok=True)
        
        # 결합된 이미지 저장
        vutils.save_image(comparison.cpu(),
                          os.path.join(save_dir, "reconstructed_vs_original.png"),
                          normalize=True,
                          nrow=1)  # 각 행에 한 이미지씩 저장 (원본-재구성 이미지 쌍이 한 행에 나열)

        print(f"Images saved to {save_dir}/reconstructed_vs_original.png")