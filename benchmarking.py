import os
import torch
from hiera.hiera import Hiera
from hiera.benchmarking import benchmark

def main():
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Model configuration
    MODEL_BACKBONE = "hiera-2d"  # Options: 'hiera-2d' or 'hiera-3d'
    BATCH_SIZE = 64
    SCREEN_CHUNK_SIZE = 30
    SCREEN_SAMPLING_RATE = 30
    RESPONSE_CHUNK_SIZE = 8
    RESPONSE_SAMPLING_RATE = 8
    BEHAVIOR_AS_CHANNELS = True
    REPLACE_NANS_WITH_MEANS = True
    DIM_HEAD = 64
    NUM_HEADS = 2
    DROP_PATH_RATE = 0
    MLP_RATIO = 4

    # Set video size and input channels based on model backbone
    if MODEL_BACKBONE == "hiera-2d":
        VIDEO_SIZE = [32, 64]
        IN_CHANNELS = 3
    elif MODEL_BACKBONE == "hiera-3d":
        VIDEO_SIZE = [32, 64]
        IN_CHANNELS = 6
    else:
        raise ValueError("Invalid model backbone specified")

    # Initialize Hiera model
    if MODEL_BACKBONE == "hiera-2d":
        model = Hiera(
            input_size=(VIDEO_SIZE[0], VIDEO_SIZE[1]),
            num_heads=1,
            embed_dim=96,
            stages=(2, 1),  # 3 transformer layers
            q_pool=1,
            in_chans=IN_CHANNELS,
            q_stride=(1, 1),
            mask_unit_size=(8, 8),
            patch_kernel=(5, 5),
            patch_stride=(2, 2),
            patch_padding=(2, 2),
            sep_pos_embed=False,
            drop_path_rate=DROP_PATH_RATE,
            mlp_ratio=MLP_RATIO,
        )
    elif MODEL_BACKBONE == "hiera-3d":
        model = Hiera(
            input_size=(SCREEN_CHUNK_SIZE, VIDEO_SIZE[0], VIDEO_SIZE[1]),
            num_heads=3,
            embed_dim=96,
            stages=(2, 1),  # 3 transformer layers
            q_pool=1,
            in_chans=IN_CHANNELS,
            q_stride=(1, 1, 1),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(5, 5, 5),
            patch_stride=(3, 2, 2),
            patch_padding=(1, 2, 2),
            sep_pos_embed=True,
            drop_path_rate=DROP_PATH_RATE,
            mlp_ratio=MLP_RATIO,
        )

    # Move model to GPU and set precision
    model = model.cuda().to(torch.float32)

    # Create example input
    if MODEL_BACKBONE == "hiera-2d":
        example_input = torch.ones(8, IN_CHANNELS, VIDEO_SIZE[0], VIDEO_SIZE[1], device="cuda", dtype=torch.float32)
    elif MODEL_BACKBONE == "hiera-3d":
        example_input = torch.ones(8, IN_CHANNELS, SCREEN_CHUNK_SIZE, VIDEO_SIZE[0], VIDEO_SIZE[1], device="cuda", dtype=torch.float32)

    # Forward pass
    output = model(example_input, return_intermediates=True)
    hiera_output = output[-1][-1]
    print("Output shape:", hiera_output.shape)  # Expected: (b, t, h, w, c)

    # Benchmark the model
    if MODEL_BACKBONE == "hiera-2d":
        benchmark(
            model=model,
            device=0,
            input_size=(IN_CHANNELS, VIDEO_SIZE[0], VIDEO_SIZE[1]),
            batch_size=BATCH_SIZE,
            runs=100,
            use_fp16=True,
            verbose=True,
        )
    elif MODEL_BACKBONE == "hiera-3d":
        benchmark(
            model=model,
            device=0,
            input_size=(IN_CHANNELS, SCREEN_CHUNK_SIZE, VIDEO_SIZE[0], VIDEO_SIZE[1]),
            batch_size=BATCH_SIZE,
            runs=100,
            use_fp16=True,
            verbose=True,
        )

if __name__ == "__main__":
    main()
