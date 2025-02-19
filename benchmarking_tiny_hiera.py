import os
import torch
from hiera.hiera import Hiera
from hiera.benchmarking import benchmark


def main():
    """Main function to initialize and benchmark the Hiera model."""
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Model configuration
    model_backbone = "hiera-2d"  # Options: 'hiera-2d' or 'hiera-3d'
    batch_size = 64
    screen_chunk_size = 30
    screen_sampling_rate = 30
    response_chunk_size = 8
    response_sampling_rate = 8
    behavior_as_channels = True
    replace_nans_with_means = True
    dim_head = 64
    num_heads = 2
    drop_path_rate = 0
    mlp_ratio = 4

    # Set video size and input channels based on model backbone
    if model_backbone == "hiera-2d":
        video_size = [32, 64]
        in_channels = 3
    elif model_backbone == "hiera-3d":
        video_size = [32, 64]
        in_channels = 6
    else:
        raise ValueError("Invalid model backbone specified")

    # Initialize Hiera model
    if model_backbone == "hiera-2d":
        model = Hiera(
            input_size=(video_size[0], video_size[1]),
            num_heads=1,
            embed_dim=96,
            stages=(2, 1),  # 3 transformer layers
            q_pool=1,
            in_chans=in_channels,
            q_stride=(1, 1),
            mask_unit_size=(8, 8),
            patch_kernel=(5, 5),
            patch_stride=(2, 2),
            patch_padding=(2, 2),
            sep_pos_embed=False,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
        )
    else:  # model_backbone == "hiera-3d"
        model = Hiera(
            input_size=(screen_chunk_size, video_size[0], video_size[1]),
            num_heads=3,
            embed_dim=96,
            stages=(2, 1),  # 3 transformer layers
            q_pool=1,
            in_chans=in_channels,
            q_stride=(1, 1, 1),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(5, 5, 5),
            patch_stride=(3, 2, 2),
            patch_padding=(1, 2, 2),
            sep_pos_embed=True,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
        )

    # Move model to GPU and set precision
    model = model.cuda().to(torch.float32)

    # Create example input
    if model_backbone == "hiera-2d":
        example_input = torch.ones(
            8, in_channels, video_size[0], video_size[1],
            device="cuda", dtype=torch.float32
        )
    else:  # model_backbone == "hiera-3d"
        example_input = torch.ones(
            8, in_channels, screen_chunk_size, video_size[0], video_size[1],
            device="cuda", dtype=torch.float32
        )

    # Forward pass
    output = model(example_input, return_intermediates=True)
    hiera_output = output[-1][-1]
    print("Output shape:", hiera_output.shape)  # Expected: (b, t, h, w, c)

    # Benchmark the model
    benchmark(
        model=model,
        device=0,
        input_size=(in_channels, *([screen_chunk_size] if model_backbone == "hiera-3d" else []), video_size[0], video_size[1]),
        batch_size=batch_size,
        runs=100,
        use_fp16=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()