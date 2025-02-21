import os
from hiera.benchmarking import tiny_mouse_hiera_benchmark

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"

if __name__ == "__main__":
    tiny_mouse_hiera_benchmark(force_fa=True)