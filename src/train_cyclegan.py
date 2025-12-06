from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from .datasets import UnpairedImageDataset
from .models import NLayerDiscriminator, ResnetGenerator, init_weights
from .utils import get_device, save_sample_grid


class ReplayBuffer:
    """
    Stores previously generated images to reduce model oscillation.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, tensors: torch.Tensor) -> torch.Tensor:
        result = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(tensor)
                result.append(tensor)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, len(self.data), (1,)).item()
                    result.append(self.data[idx].clone())
                    self.data[idx] = tensor
                else:
                    result.append(tensor)
        return torch.cat(result, dim=0)


def gan_criterion(use_lsgan: bool = True) -> nn.Module:
    return nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()


def cycle_gan_step(
    batch: Tuple[torch.Tensor, torch.Tensor],
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    criterions: Dict[str, nn.Module],
    buffers: Dict[str, ReplayBuffer],
    lambdas: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    real_mri, real_ct = batch
    real_mri = real_mri.to(device)
    real_ct = real_ct.to(device)

    # Modified: Dynamic discriminator target tensor shapes instead of hard-coded (1,30,30)
    G_AB = models["G_AB"]
    G_BA = models["G_BA"]
    D_A = models["D_A"]
    D_B = models["D_B"]
    
    # Compute validity maps dynamically using discriminator output shapes
    # create target labels using full batch prediction shape
    pred_real_A = D_A(real_mri)
    valid_A = torch.ones_like(pred_real_A)
    fake_A  = torch.zeros_like(pred_real_A)

    pred_real_B = D_B(real_ct)
    valid_B = torch.ones_like(pred_real_B)
    fake_B  = torch.zeros_like(pred_real_B)


    # ------------------
    #  Train Generators
    # ------------------
    optimizers["G"].zero_grad()

    # Identity loss
    id_ct = G_AB(real_ct)
    loss_id_ct = criterions["L1"](id_ct, real_ct)
    id_mri = G_BA(real_mri)
    loss_id_mri = criterions["L1"](id_mri, real_mri)
    loss_identity = (loss_id_ct + loss_id_mri) * lambdas["identity"]

    # GAN loss - Modified: Use valid_B/fake_B for CT loss and valid_A/fake_A for MRI loss
    fake_ct = G_AB(real_mri)
    pred_fake_ct = D_B(fake_ct)
    valid_B = torch.ones_like(pred_fake_ct)  # match shape!
    loss_G_AB = criterions["GAN"](pred_fake_ct, valid_B)


    fake_mri = G_BA(real_ct)
    pred_fake_mri = D_A(fake_mri)
    valid_A = torch.ones_like(pred_fake_mri)
    loss_G_BA = criterions["GAN"](pred_fake_mri, valid_A)


    # Cycle loss
    rec_mri = G_BA(fake_ct)
    rec_ct = G_AB(fake_mri)
    loss_cycle_mri = criterions["L1"](rec_mri, real_mri)
    loss_cycle_ct = criterions["L1"](rec_ct, real_ct)
    loss_cycle = (loss_cycle_mri + loss_cycle_ct) * lambdas["cycle"]

    loss_G = loss_identity + loss_G_AB + loss_G_BA + loss_cycle
    loss_G.backward()
    optimizers["G"].step()

    # -----------------------
    #  Train Discriminator A (MRI)
    # -----------------------
    optimizers["D_A"].zero_grad()

    # Real MRI
    pred_real_A = D_A(real_mri)
    valid_A = torch.ones_like(pred_real_A)
    loss_real_A = criterions["GAN"](pred_real_A, valid_A)

    # Fake MRI from generator
    fake_buffered_A = buffers["A"].push_and_pop(fake_mri.detach())  # replay buffer
    pred_fake_A = D_A(fake_buffered_A)
    fake_A = torch.zeros_like(pred_fake_A)
    loss_fake_A = criterions["GAN"](pred_fake_A, fake_A)

    loss_D_A = 0.5 * (loss_real_A + loss_fake_A)
    loss_D_A.backward()
    optimizers["D_A"].step()


    # -----------------------
    #  Train Discriminator B (CT)
    # -----------------------
    optimizers["D_B"].zero_grad()

    # Real CT
    pred_real_B = D_B(real_ct)
    valid_B = torch.ones_like(pred_real_B)
    loss_real_B = criterions["GAN"](pred_real_B, valid_B)

    # Fake CT from generator
    fake_buffered_B = buffers["B"].push_and_pop(fake_ct.detach())
    pred_fake_B = D_B(fake_buffered_B)
    fake_B = torch.zeros_like(pred_fake_B)
    loss_fake_B = criterions["GAN"](pred_fake_B, fake_B)

    loss_D_B = 0.5 * (loss_real_B + loss_fake_B)
    loss_D_B.backward()
    optimizers["D_B"].step()


    return {
        "loss_G": loss_G.item(),
        "loss_G_identity": loss_identity.item(),
        "loss_G_GAN": (loss_G_AB + loss_G_BA).item(),
        "loss_G_cycle": loss_cycle.item(),
        "loss_D_A": loss_D_A.item(),
        "loss_D_B": loss_D_B.item(),
    }


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: Dict) -> DataLoader:
    # Modified: Added basic image augmentations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = UnpairedImageDataset(
        mri_dir=Path(cfg["data"]["mri_dir"]),
        ct_dir=Path(cfg["data"]["ct_dir"]),
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=True,
    )


def save_checkpoint(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    path: Path,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "G_AB": models["G_AB"].state_dict(),
            "G_BA": models["G_BA"].state_dict(),
            "D_A": models["D_A"].state_dict(),
            "D_B": models["D_B"].state_dict(),
            "opt_G": optimizers["G"].state_dict(),
            "opt_D_A": optimizers["D_A"].state_dict(),
            "opt_D_B": optimizers["D_B"].state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    models["G_AB"].load_state_dict(ckpt["G_AB"])
    models["G_BA"].load_state_dict(ckpt["G_BA"])
    models["D_A"].load_state_dict(ckpt["D_A"])
    models["D_B"].load_state_dict(ckpt["D_B"])
    optimizers["G"].load_state_dict(ckpt["opt_G"])
    optimizers["D_A"].load_state_dict(ckpt["opt_D_A"])
    optimizers["D_B"].load_state_dict(ckpt["opt_D_B"])
    start_epoch = int(ckpt.get("epoch", 1))
    start_step = int(ckpt.get("step", 0))
    return start_epoch, start_step


def train(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    # Modified: Enable GPU training safely - do not force CPU by default
    device = get_device(force_cpu=cfg["train"].get("force_cpu", False))

    dataloader = build_dataloader(cfg)

    # Models
    G_AB = ResnetGenerator(n_blocks=cfg["model"].get("n_res_blocks", 6))
    G_BA = ResnetGenerator(n_blocks=cfg["model"].get("n_res_blocks", 6))
    D_A = NLayerDiscriminator()
    D_B = NLayerDiscriminator()
    for net in (G_AB, G_BA, D_A, D_B):
        init_weights(net, "normal", 0.02)
        net.to(device)

    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=cfg["train"]["lr"],
        betas=(0.5, 0.999),
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=cfg["train"]["lr"], betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=cfg["train"]["lr"], betas=(0.5, 0.999))

    models = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}
    optimizers = {"G": optimizer_G, "D_A": optimizer_D_A, "D_B": optimizer_D_B}
    buffers = {"A": ReplayBuffer(), "B": ReplayBuffer()}
    criterions = {"GAN": gan_criterion(cfg["train"].get("lsgan", True)), "L1": nn.L1Loss()}
    lambdas = {"cycle": cfg["train"]["lambda_cycle"], "identity": cfg["train"]["lambda_identity"]}

    out_dir = Path(cfg["train"].get("save_dir", "runs/cyclegan"))
    sample_dir = out_dir / "samples"

    # Resume logic
    resume_path: Optional[str] = cfg["train"].get("resume_checkpoint")
    start_epoch = 1
    total_steps = 0
    if resume_path:
        ckpt_path = Path(resume_path)
        if ckpt_path.exists():
            start_epoch, total_steps = load_checkpoint(ckpt_path, models, optimizers)
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}, step {total_steps}")
        else:
            print(f"Resume checkpoint {ckpt_path} not found; starting fresh.")

    save_every_steps = cfg["train"].get("save_every_steps", 50)
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        epoch_start = time.time()
        for batch in dataloader:
            total_steps += 1
            losses = cycle_gan_step(batch, models, optimizers, criterions, buffers, lambdas, device)

            if total_steps % cfg["train"].get("log_every", 25) == 0:
                print(
                    f"[Epoch {epoch}/{cfg['train']['epochs']}] "
                    f"step {total_steps} "
                    + " ".join(f"{k}: {v:.3f}" for k, v in losses.items())
                )

            if total_steps % cfg["train"].get("sample_every", 100) == 0:
                with torch.no_grad():
                    real_mri, real_ct = batch
                    real_mri = real_mri.to(device)
                    real_ct = real_ct.to(device)
                    fake_ct = G_AB(real_mri)
                    fake_mri = G_BA(real_ct)
                    rec_mri = G_BA(fake_ct)
                    rec_ct = G_AB(fake_mri)
                    imgs = [
                        real_mri[0].cpu(),
                        fake_ct[0].cpu(),
                        rec_mri[0].cpu(),
                        real_ct[0].cpu(),
                        fake_mri[0].cpu(),
                        rec_ct[0].cpu(),
                    ]
                    save_sample_grid(imgs, sample_dir / f"step_{total_steps:06d}.png", nrow=3)

            if total_steps % save_every_steps == 0:
                step_ckpt = out_dir / "checkpoints" / f"step_{total_steps:06d}.pt"
                save_checkpoint(models, optimizers, step_ckpt, epoch, total_steps)

        ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        save_checkpoint(models, optimizers, ckpt_path, epoch, total_steps)
        print(f"Epoch {epoch} done in {time.time() - epoch_start:.1f}s, saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN for MRI->CT synthesis.")
    parser.add_argument("--config", type=Path, default=Path("configs/cyclegan_cpu.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
