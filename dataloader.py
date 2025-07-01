from torch.utils.data import Dataset
import itertools, logging, random
from pathlib import Path
from PIL.ImageOps import exif_transpose
from diffusers.training_utils import find_nearest_bucket, parse_buckets_string
from torchvision import transforms
from torchvision.transforms.functional import crop
import torch


class FluxKontext(Dataset):
    """
    Dataset that yields (source_image, target_image, prompt) triplets
    ready for Flux-Kontext edit fine-tuning.
    """

    def __init__(
        self,
        args,
        split="train",
        repeats: int = 1,
        center_crop: bool = False,
    ):
        """
        Expected `args` attributes
        -------------------------
        dataset_name, dataset_config_name, cache_dir
        source_image_column, target_image_column, caption_column
        resolution, aspect_ratio_buckets, random_flip
        """
        # ---------------------------------------------------------------------
        # 0.  Load HF dataset
        # ---------------------------------------------------------------------
        if args.dataset_name is None:
            raise ValueError("--dataset_name is required")

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("`datasets` library missing: pip install datasets") from e

        ds = load_dataset(args.dataset_name,
                          args.dataset_config_name,
                          cache_dir=args.cache_dir)[split]
        
        cols = ds.column_names

        # column names --------------------------------------------------------
        src_col  = args.source_image_column
        tgt_col  = args.target_image_column
        cap_col  = args.caption_column

        if src_col not in cols or tgt_col not in cols:
            raise ValueError(
                f"Dataset columns are {cols}. "
                f"Required --source_image_column ({src_col}) "
                f"and --target_image_column ({tgt_col}) not both present."
            )

        self.src_imgs = ds[src_col]
        self.tgt_imgs = ds[tgt_col]

        if cap_col and cap_col not in cols:
            raise ValueError(
                f"--caption_column '{cap_col}' not in dataset columns {cols}"
            )
        self.captions = ds[cap_col] if cap_col else None
        self.instance_prompt = args.instance_prompt  # default prompt text

        # ---------------------------------------------------------------------
        # 1.  Bucket setup (same as DreamBooth script)
        # ---------------------------------------------------------------------
        self.buckets = (parse_buckets_string(args.aspect_ratio_buckets)
                        if args.aspect_ratio_buckets
                        else [(args.resolution, args.resolution)])
        logging.info(f"Using aspect-ratio buckets: {self.buckets}")

        # ---------------------------------------------------------------------
        # 2.  Pre-process every pair once so __getitem__ is cheap
        # ---------------------------------------------------------------------
        self.src_pixel_values, self.tgt_pixel_values, self.bucket_ids = [], [], []

        # build transforms ***once*** per dataset ─ identical params for both imgs
        for idx, (src_pil, tgt_pil) in enumerate(zip(self.src_imgs, self.tgt_imgs)):
            # repeat K times for class-balancing
            for _ in range(repeats):
                src_pil = exif_transpose(src_pil)
                tgt_pil = exif_transpose(tgt_pil)

                if src_pil.mode != "RGB":
                    src_pil = src_pil.convert("RGB")
                if tgt_pil.mode != "RGB":
                    tgt_pil = tgt_pil.convert("RGB")

                # choose bucket by *target* image (any strategy is OK as long as consistent)
                h, w = tgt_pil.size[1], tgt_pil.size[0]
                bucket_idx = find_nearest_bucket(h, w, self.buckets)
                tgt_h, tgt_w = self.buckets[bucket_idx]
                size = (tgt_h, tgt_w)

                # record so we can debug later
                self.bucket_ids.append(bucket_idx)

                resize = transforms.Resize(size, transforms.InterpolationMode.BILINEAR)
                crop_fn = (transforms.CenterCrop(size) if center_crop
                           else transforms.RandomCrop(size))
                flip_fn = transforms.RandomHorizontalFlip(p=1.0)
                to_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])

                def process(img):
                    img = resize(img)
                    if center_crop:
                        img = crop_fn(img)
                    else:
                        y1, x1, h_, w_ = crop_fn.get_params(img, size)
                        img = crop(img, y1, x1, h_, w_)
                    if args.random_flip and random.random() < 0.5:
                        img = flip_fn(img)
                    return to_tensor(img)

                self.src_pixel_values.append(process(src_pil))
                self.tgt_pixel_values.append(process(tgt_pil))

        # ---------------------------------------------------------------------
        # 3.  Optional caption expansion
        # ---------------------------------------------------------------------
        if self.captions is not None:
            prompts_expanded = []
            for cap in self.captions:
                prompts_expanded.extend(itertools.repeat(cap, repeats))
            self.prompts = prompts_expanded
        else:
            self.prompts = None

        self.length = len(self.src_pixel_values)
        assert self.length == len(self.tgt_pixel_values)

    # =======================================================================
    # PyTorch Dataset API
    # =======================================================================
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        ex = {
            "src_pixels": self.src_pixel_values[idx],
            "tgt_pixels": self.tgt_pixel_values[idx],
            "bucket_idx": self.bucket_ids[idx],
        }

        if self.prompts is not None:
            ex["prompt"] = self.prompts[idx] if self.prompts[idx] else self.instance_prompt
        else:
            ex["prompt"] = self.instance_prompt

        return ex


# ------------------------------------------------------------------------------
# Collate -- stacks source & target tensors separately
# ------------------------------------------------------------------------------
def collate_fn(examples):
    src = torch.stack([e["src_pixels"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    tgt = torch.stack([e["tgt_pixels"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    prompts = [e["prompt"] for e in examples]

    batch = {"src_pixels": src,
            "tgt_pixels": tgt,
            "prompts": prompts}

    return batch













# ------------------------------------------------------------------------------
#  test dataloader
# ------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1.  Minimal arg-parser with just the fields the Dataset needs
# ---------------------------------------------------------------------
import argparse
from torchvision.utils import save_image
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="raresense/Pose_Flatlay_Test")
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument("--cache_dir", default=None)

    p.add_argument("--source_image_column", default="source")
    p.add_argument("--target_image_column", default="target")
    p.add_argument("--caption_column", default="ai_name")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--aspect_ratio_buckets", default="1184,880")
    p.add_argument("--random_flip", action="store_true")

    p.add_argument("--instance_prompt", default="photo")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--center_crop", action="store_true")
    return p.parse_args()
# ---------------------------------------------------------------------
def main():

    args = get_args()

    # -----------------------------------------------------------------
    # 2.  Build dataset & take first five items
    # -----------------------------------------------------------------
    ds = FluxKontext(args=args, split="train",
                     repeats=args.repeats,
                     center_crop=args.center_crop)

    print(f"Dataset length: {len(ds)}")

    idxs   = list(range(min(5, len(ds))))
    sample = collate_fn([ds[i] for i in idxs])        # batch of 5

    print("src_pixels shape:", sample["src_pixels"].shape)
    print("tgt_pixels shape:", sample["tgt_pixels"].shape)
    print("prompts      :", sample["prompts"])

    # -----------------------------------------------------------------
    # 3.  Save original + processed images side-by-side
    # -----------------------------------------------------------------
    outdir = Path("debug_samples")
    outdir.mkdir(exist_ok=True)

    to_pil = transforms.ToPILImage()

    for k, i in enumerate(idxs):
        # originals are in ds.src_imgs / ds.tgt_imgs
        ds.src_imgs[i].save(outdir / f"sample-{k}_src_orig.png")
        ds.tgt_imgs[i].save(outdir / f"sample-{k}_tgt_orig.png")

        # processed tensors are in the collated batch (-1…1 range)
        #   → convert back to [0,1] and save
        src_proc = (sample["src_pixels"][k] * 0.5 + 0.5).clamp(0, 1)
        tgt_proc = (sample["tgt_pixels"][k] * 0.5 + 0.5).clamp(0, 1)
        save_image(src_proc, outdir / f"sample-{k}_src_proc.png")
        save_image(tgt_proc, outdir / f"sample-{k}_tgt_proc.png")

    print(f"Images written to {outdir.resolve()}")

if __name__ == "__main__":
    main()