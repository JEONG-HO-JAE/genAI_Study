from omegaconf import OmegaConf, DictConfig, open_dict
from .respace import SpacedDiffusion, space_timesteps
from . import gaussian_diffusion as gd
from .unet import SuperResModel, UNetModel, EncoderUNetModel
NUM_CLASSES = 1000


def setup_config(config: DictConfig) -> DictConfig:
    """
    Add default values to the config if missing.
    """
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        warmup_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        learn_sigma=False,
    )
    # print("\n" + "="*50)
    # print("[SETUP CONFIG] Setting up default configurations...")
    OmegaConf.set_struct(config, True)
    for key, value in defaults.items():
        if key in config:
            # print(f"  - [SKIP] '{key}' already exists in config.")
            continue  # Skip if key exists at the top level
        if any(key in nested for nested in config.values() if isinstance(nested, DictConfig)):
            # print(f"  - [SKIP] '{key}' exists in nested config.")
            continue
        with open_dict(config):
            # print(f"  - [ADD] '{key}' added to config.")
            config[key] = value
    # print("[SETUP CONFIG] Defaults setup complete.")
    # print("="*50 + "\n")
    
    config = set_model_defaults(config)
    return config


def set_model_defaults(config):
    unet_defaults = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    diffusion_defaults = dict(    
        steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    # print("\n" + "="*50)
    # print("[SETUP MODEL] Setting up model configurations...")
    OmegaConf.set_struct(config, True)
    for key, value in unet_defaults.items():
        if key in config.model:
            # print(f"  - [SKIP] '{key}' already exists in model config.")
            continue
        else:
            with open_dict(config):
                # print(f"  - [ADD] '{key}' added to model config.")
                config.model[key] = value
    # print("[SETUP MODEL] Model defaults setup complete.")
    # print("="*50 + "\n")
    
    # print("\n" + "="*50)
    # print("[SETUP DIFFUSION] Setting up DIFFUSION configurations...")
    OmegaConf.set_struct(config, True)
    for key, value in diffusion_defaults.items():
        if key in config:
            # print(f"  - [SKIP] '{key}' already exists in config.")
            continue  # Skip if key exists at the top level
        if key in config.diffusion:
            # print(f"  - [SKIP] '{key}' already exists in DIFFUSION config.")
            continue
        else:
            with open_dict(config):
                # print(f"  - [ADD] '{key}' added to DIFFUSION config.")
                config.diffusion[key] = value
    # print("[SETUP DIFFUSION] DIFFUSION defaults setup complete.")
    # print("="*50 + "\n")
    
    return config


def create_model_and_diffusion(config):
    return create_model(config), create_diffusion(config)


def create_model(config):
    """
    Create and configure the model based on the provided config.
    """
    # Fetch necessary values from config
    image_size = config.model.get("image_size", 64)  # Default to 64 if not specified
    channel_mult = config.model.get("channel_mult", "")
    attention_resolutions = config.model.get("attention_resolutions", "16,8")
    
    # Resolve channel multiplier
    if not channel_mult:  # If channel_mult is not explicitly defined
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
    else:
        # Convert channel_mult from string to tuple
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # Resolve attention resolutions
    try:
        attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]
    except ValueError:
        raise ValueError(f"Invalid attention_resolutions format: {attention_resolutions}")

    # Update model configuration
    with open_dict(config.model):  # Allow updates to config.model
        config.model.channel_mult = channel_mult
        config.model.attention_ds = attention_ds

    # print("\n" + "="*50)
    # print("[CREATE MODEL] Model configurations set:")
    # print(OmegaConf.to_yaml(config.model))
    # print("="*50 + "\n")
    
    # Instantiate the model using Hydra
    return UNetModel(
        image_size=config.model.image_size,
        in_channels=config.dataset.channels,
        model_channels=config.model.num_channels,
        out_channels=(3 if not config.learn_sigma else 6),
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=tuple(config.model.attention_ds),
        dropout=config.model.dropout,
        channel_mult=config.model.channel_mult,
        num_classes=(NUM_CLASSES if config.model.class_cond else None),
        use_checkpoint=config.model.use_checkpoint,
        use_fp16=config.model.use_fp16,
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
        num_heads_upsample=config.model.num_heads_upsample,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
        resblock_updown=config.model.resblock_updown,
        use_new_attention_order=config.model.use_new_attention_order,
    )


def create_diffusion(config):
    betas = gd.get_named_beta_schedule(config.diffusion.noise_schedule, config.diffusion.noise_steps)
    if config.diffusion.use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif config.diffusion.rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not config.diffusion.timestep_respacing:
        timestep_respacing = [config.steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(config.steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not config.diffusion.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not config.diffusion.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not config.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=config.diffusion.rescale_timesteps,
    )