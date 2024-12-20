"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import random

from DDPM.image_datasets import load_data
from DDPM import dist_util, logger
from DDPM.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    
    # Set random seed
    seed = 42
    random.seed(seed)
    th.manual_seed(seed)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    batch = next(data)
    images, _ = batch
    images = images.to(dist_util.dev())

    # 저장 경로 설정
    out_path = os.path.join(logger.get_dir(), "org_img.npz")

    # images 데이터를 numpy 배열로 변환 후 저장
    images_np = images.cpu().numpy()  # Tensor -> Numpy
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    np.savez(out_path, images=images_np)
    logger.log(f"Original images saved to {out_path}")
    
    if isinstance(args.timestep_respacing, str):
        if args.timestep_respacing.startswith("ddim"):
            desired_count = int(args.timestep_respacing[len("ddim") :])
        else:
            desired_count = int(args.timestep_respacing)
            
    # Set timesteps and x_T based on arguments
    timesteps = []
    x_T = []
    if args.check_all_loss_q:
        # Calculate timesteps based on the desired_count
        step_intervals = [0.25, 0.5, 0.75, 1.0]
        for i, ratio in enumerate(step_intervals):
            timestep_value = int(desired_count * ratio) - 1
            timestep = th.tensor(timestep_value, device=images.device)
            timesteps.append(timestep)
            x_T.append(diffusion.q_sample(images, timestep))
        logger.log(f"Calculated timesteps: {timesteps}")
    else:
        x_T = [None]
    logger.log("==================sampling=====================")

    for i, xt in enumerate(x_T):  # Loop through timesteps and their corresponding noisy samples
        all_images = []  # 각 타임스텝마다 초기화
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            if args.check_all_loss_q is False:
                timestep = None
            else:
                timestep = timesteps[i]
                
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                xt,  # Use the corresponding x_T
                timestep,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=args.progress
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            # Gather samples across all processes
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            logger.log(f"created {len(all_images) * args.batch_size} samples")

        # Save samples and x_T after each timestep
        save_samples(all_images, all_labels, args, i, xt)

    dist.barrier()
    logger.log("sampling complete")


def save_samples(all_images, all_labels, args, timestep_index, xt):
    """
    Save the generated samples and corresponding x_T to a file.
    """
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    xt_np = xt.cpu().numpy()  # Convert x_T to numpy
    xt_np = np.transpose(xt_np, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_loss_{3-timestep_index}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr=arr, labels=label_arr, x_T=xt_np)
        else:
            np.savez(out_path, arr=arr, x_T=xt_np)


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=64,
        batch_size=64,
        use_ddim=False,
        model_path="",
        check_all_loss_q=False,
        diffusion_steps=1000,
        progress=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()