# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm

from hiera.hiera import Hiera

# From https://github.com/facebookresearch/ToMe/
def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model e.g., (ch, h, w) or (ch, t, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput

def tiny_mouse_hiera_benchmark(
    model_backbone = "hiera-2d",
    batch_size = 64,
    screen_chunk_size = 30,
    screen_sampling_rate = 30,
    response_chunk_size = 8,
    response_sampling_rate = 8,
    behavior_as_channels = True,
    replace_nans_with_means = True,
    dim_head = 64,
    num_heads = 2,
    drop_path_rate = 0,
    mlp_ratio = 4,
    force_fa = True):

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
            force_fa=force_fa
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
            force_fa=force_fa
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
