"""
ddp_inference.py
----------------
A script to perform DDP inference with a Hugging Face model.
This script demonstrates how to use DDP with a custom sampler
to perform inference on a dataset of images, saving embeddings
to .npz files alongside the original images.

Usage:
  torchrun --nproc_per_node=2 \
    ddp_inference.py \
      --world_size=2 \
      --data_dir /path/to/images \
      --batch_size 4 \
      --disable_transform \
      --use_processor \
      --use_pooler \
      --output_dir /path/to/save/npz
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


from evfr.datasets.image_folder import ImageFolderDataset
from evfr.utils.custom_ddp_sampler import CustomDistributedSampler
from evfr.embeddings.hf_dinov2 import HuggingFaceDinoV2Embedding

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Inference with Hugging Face model.")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes.")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with images.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional directory to save embeddings. If None, saves alongside images.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_pooler", action="store_true", help="Use pooler output if available.")
    parser.add_argument("--use_processor", action="store_true", help="Use HF AutoImageProcessor.")
    parser.add_argument("--disable_transform", action="store_true",
                        help="Disable dataset transforms (return raw PIL).")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level: DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # 1. Initialize process group
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank= local_rank,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"[Rank {rank}] Initialized process group: world_size={world_size}, local_rank={local_rank}")

    # 2. Create DataLoader with CustomDistributedSampler
    dataset = ImageFolderDataset(
        root_dir=args.data_dir,
        transform=None,
        disable_transform=args.disable_transform
    )
    sampler = CustomDistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    def collate_fn(batch):
        # batch = [(image, idx), (image, idx), ...]
        # we transpose -> ([image, image, ...], [idx, idx, ...])
        # but we also need the original dataset paths to save .npz
        # => We'll fetch them from dataset.image_paths for each idx.
        imgs, indices = list(zip(*batch))
        return imgs, indices

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    logger.info(f"[Rank {rank}] DataLoader created with {len(dataloader)} batches.")

    # 3. Initialize model and wrap in DDP
    model = HuggingFaceDinoV2Embedding(
        use_pooler=args.use_pooler,
        use_processor=args.use_processor,
        device=f"cuda:{local_rank}" if local_rank >= 0 else "cpu"
    )
    model.model = DDP(
        model.model,
        device_ids=[local_rank] if local_rank >= 0 else None,
        output_device=local_rank if local_rank >= 0 else None
    )

    logger.info(f"[Rank {rank}] Model wrapped in DDP. Using device cuda:{local_rank}" 
                if local_rank >= 0 else f"[Rank {rank}] Model on CPU")

    # If user specified an output_dir, create it (only once per rank, ignoring collisions)
    if args.output_dir is not None and rank >= 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"[Rank {rank}] Output directory set to: {args.output_dir}")

    local_processed_count = 0
    logger.info(f"[Rank {rank}] Starting inference...")


    # 4. Perform inference and save embeddings
    with torch.no_grad():
        for batch_idx, (imgs, indices) in enumerate(dataloader):
            # (Optional) debug-level log for each batch
            logger.debug(f"[Rank {rank}] Processing batch {batch_idx}/{len(dataloader)} with size={len(imgs)}")

            if model.use_processor:
                embeddings = model.extract_batch(imgs, batch_size=len(imgs))
            else:
                embeddings = model.extract_batch(imgs, batch_size=len(imgs))

            for i, emb in zip(indices, embeddings):
                image_path = dataset.image_paths[i]

                if args.output_dir is None:
                    # Save alongside original images
                    base, _ = os.path.splitext(image_path)
                    out_path = base + ".npz"
                else:
                    # Save to separate dir, preserving filename
                    filename = os.path.basename(image_path)
                    base, _ = os.path.splitext(filename)
                    out_path = os.path.join(args.output_dir, base + ".npz")

                np.savez_compressed(out_path, embedding=emb)
                local_processed_count += 1

    logger.info(f"[Rank {rank}] Finished inference. Processed {local_processed_count} images total.")

    # 5. Gather counts from all ranks
    local_count_tensor = torch.tensor([local_processed_count], dtype=torch.long, 
                                      device=f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
    gathered_counts = [torch.zeros(1, dtype=torch.long, 
                      device=f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
                       for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count_tensor)

    dist.barrier()
    dist.destroy_process_group()

    # 6. Log and cleanup
    if rank == 0:
        logger.info("[Rank 0] Inference complete. Each rank wrote .npz files for its samples.")
        for r, cnt in enumerate(gathered_counts):
            logger.info(f"  Rank {r} processed {cnt.item()} images.")
        if args.output_dir:
            logger.info(f"All embeddings saved under: {args.output_dir}")


if __name__ == "__main__":
    main()