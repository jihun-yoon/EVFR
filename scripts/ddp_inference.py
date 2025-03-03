"""
ddp_inference.py
----------------
Example script demonstrating how to perform multi-GPU inference
using DistributedDataParallel (DDP) with the EVFR library.
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# Example: We'll use the HuggingFace Swin embedding from your library
from evfr.embeddings.hf_dinov2 import HuggingFaceDinoV2Embedding
from evfr.sampling.fps import FarthestPointSampler
from evfr.utils.distance_metrics import euclidean_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per process"
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Total number of processes"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="nccl", help="Distributed backend"
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default="env://",
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=16,
        help="Number of dummy images for testing",
    )
    parser.add_argument(
        "--use_pooler", action="store_true", help="Use pooler output if available"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. Initialize process group (torchrun sets LOCAL_RANK, RANK, WORLD_SIZE automatically in env)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=int(os.environ.get("RANK", 0)),
    )

    # 2. Create model (each process has its own model replica)
    #    We'll place the model on the current GPU and wrap it in DDP.
    model = HuggingFaceDinoV2Embedding(
        use_pooler=args.use_pooler, device=f"cuda:{local_rank}"
    )
    # Wrap the model for DDP usage
    model.model = DDP(model.model, device_ids=[local_rank], output_device=local_rank)
    # Note: We wrap the `model.model` sub-module, not the entire HuggingFaceBaseEmbedding,
    # because that's where forward pass and parameters live.

    # 3. Create dummy data for inference. In real cases, you'd load your dataset or generator.
    #    We'll just create N random frames. Each process will handle a subset.
    total_samples = args.dataset_size
    # Simple approach: each process gets its own chunk of data
    # (A more advanced approach uses DistributedSampler, etc.)
    samples_per_rank = (total_samples + args.world_size - 1) // args.world_size
    start_idx = local_rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)

    # Generate only the chunk for this rank
    frames = [np.random.rand(224, 224, 3) for _ in range(start_idx, end_idx)]

    # 4. Run batch inference
    #    We still use model.extract_batch, but now under DDP context.
    embeddings = model.extract_batch(frames, batch_size=args.batch_size)

    # 5. Optionally do some sampling or local post-processing
    #    e.g., Farthest Point Sampling on each rank's portion
    sampler = FarthestPointSampler(distance_fn=euclidean_distance)
    # If we only have a few embeddings on each rank, let's just pick k=2
    if len(embeddings) > 0:
        selected_indices = sampler.sample(embeddings, k=min(2, len(embeddings)))
    else:
        selected_indices = []

    # 6. Gather results to rank 0 or each rank writes out partial results
    #    For demonstration, we gather the selected indices across all ranks.
    selected_indices_tensor = torch.tensor(
        selected_indices, dtype=torch.long, device=f"cuda:{local_rank}"
    )
    # Let's assume each rank might produce up to 10 indices, fill with -1 if shorter
    padded = torch.full((10,), -1, dtype=torch.long, device=f"cuda:{local_rank}")
    padded[: len(selected_indices_tensor)] = selected_indices_tensor

    gathered_list = [torch.zeros_like(padded) for _ in range(args.world_size)]
    dist.all_gather(gathered_list, padded)

    # rank 0 prints the aggregated results
    if dist.get_rank() == 0:
        print("Gathered selected indices from all ranks:")
        for i, g in enumerate(gathered_list):
            valid = g[g >= 0].tolist()
            print(f"  Rank {i} -> {valid}")

    # 7. Finalize
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
