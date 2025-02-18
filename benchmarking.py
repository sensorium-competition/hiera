import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"

import torch
from hiera.hiera import Hiera
from hiera.benchmarking import benchmark

model_bkbn = 'hiera' # 'hiera' or 'hieradet'

if model_bkbn == "hiera":
    video_size = [32, 64]
    in_channels = 3
    
batchsize=64

screen_chunk_size = 30
screen_sampling_rate = 30

response_chunk_size = 8
response_sampling_rate = 8

behavior_as_channels = True
replace_nans_with_means = True

dim_head = 64
num_heads = 2
drop_path_rate = 0
mlp_ratio=4


if model_bkbn == 'hiera':
    tiny_hiera = Hiera(input_size=(video_size[0], video_size[1]),
                        num_heads=1,
                        embed_dim=96,
                        stages=(2, 1,), # 3 transformer layers 
                        q_pool=1, 
                        in_chans=in_channels,
                        q_stride=(1, 1,),
                        mask_unit_size=(8, 8),
                        patch_kernel=(5, 5),
                        patch_stride=(2, 2),
                        patch_padding=(2, 2),
                        sep_pos_embed=False, # True for 3D
                        drop_path_rate=drop_path_rate,
                        mlp_ratio=4,)

tiny_hiera = tiny_hiera.cuda().to(torch.float32);
example_input = torch.ones(8,in_channels,video_size[0], video_size[1]).to("cuda", torch.float32)
if model_bkbn == "hiera":
    out = tiny_hiera(example_input, return_intermediates=True)
elif model_bkbn == "hiera-default":
    out = tiny_hiera(example_input, return_intermediates=True)
elif model_bkbn == "hieradet":
    out = tiny_hiera(example_input)

hiera_output = out[-1][-1]
hiera_output.shape # (b, t, h, w, c): (8, 4, 9, 16, 192)
print(hiera_output.shape)

# exit()

benchmark(model=tiny_hiera, 
          device=0,
          input_size=(in_channels, video_size[0], video_size[1]),
          batch_size=batchsize,
          runs=100,
          use_fp16=True,
          verbose=True)