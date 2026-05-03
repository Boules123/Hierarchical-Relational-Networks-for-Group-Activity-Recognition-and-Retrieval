"""
Training script for person-level classifier on the Volleyball dataset.
"""

from datetime import datetime
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp.autocast_mode import autocast
from sklearn.metrics import f1_score
from configs.read_yml_cfg import get_config
from utils.logger import setup_logging
from data.dataset_loader import GroupActivityDataset, collate_fn
from utils.helpers import save_checkpoint
from models.person_model import PersonModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(cfg, model, dataloader, optimizer, criterion, device, epoch, scaler, writer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        images = batch[0].to(device) 
        labels = batch[2].to(device) 
        
        with autocast(device_type='cuda', dtype=torch.float16):
            out = model(images) 
            loss = criterion(out, labels.view(-1)) 
            loss = loss / cfg.training.grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % cfg.training.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            writer.add_scalar('Train/Loss', loss.item() * cfg.training.grad_accum_steps, epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/GradNorm', grad_norm, epoch * len(dataloader) + batch_idx)
            optimizer.zero_grad()
        
        
        batch_loss = loss.item() * cfg.training.grad_accum_steps
        total_loss += batch_loss * images.size(0)
        total_correct += (out.argmax(dim=1) == labels.view(-1)).sum().item()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.view(-1).cpu().numpy())

        
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            current_acc = total_correct / len(all_labels) if all_labels else 0
            print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {batch_loss:.4f} | Accuracy: {current_acc:.4f}")
            writer.add_scalar('Train/Accuracy', current_acc, epoch * len(dataloader) + batch_idx)
    
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(all_labels) if all_labels else 0
    writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Epoch_Accuracy', epoch_acc, epoch)
    return epoch_loss, epoch_acc

def validate(cfg, model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch[0].to(device) 
            labels = batch[2].to(device) 
            
            out = model(images) 
            loss = criterion(out, labels.view(-1)) 
            
            total_loss += loss.item() * images.size(0)
            total_correct += (out.argmax(dim=1) == labels.view(-1)).sum().item()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.view(-1).cpu().numpy())
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(all_labels) if all_labels else 0
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    writer.add_scalar('Val/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Epoch_Accuracy', epoch_acc, epoch)
    writer.add_scalar('Val/Epoch_F1', epoch_f1, epoch)
    
    print(f"Validation - Epoch [{epoch+1}/{cfg.training.num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | F1 Score: {epoch_f1:.4f}")
    
    return epoch_loss, epoch_acc, epoch_f1




def main():
    cfg = get_config("")
    set_seed(cfg.training.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(cfg.experiment.save_dir) / f"Person_classifier_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))
    logger = setup_logging(exp_dir=exp_dir)

    print(f"Device: {device} | GPUs: {num_gpus} | Seed: {cfg.training.seed} | Experiment: {cfg.experiment.exp_name} | Version: {cfg.experiment.version}")
    print(f"Experiment dir: {exp_dir}")
    
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5)
        ], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = GroupActivityDataset(
        data_dir=cfg.dataset.data_dir,
        annot_dir=cfg.dataset.annot_dir,
        split=cfg.dataset.train_split,
        transform=train_transform,
        return_person_labels=True
    )
    val_dataset = GroupActivityDataset(
        data_dir=cfg.dataset.data_dir,
        annot_dir=cfg.dataset.annot_dir,
        split=cfg.dataset.val_split,
        transform=val_transform,
        return_person_labels=True
    )
    
    effective_batch = cfg.training.batch_size * max(num_gpus, 1)
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(
        f"Batch: {cfg.training.batch_size} × {max(num_gpus, 1)} GPUs = {effective_batch} | "
        f"Accum: {cfg.training.grad_accum_steps}"
    )
    
    model = PersonModel().to(device)
    
    optimizer = model.configure_optimizers(
        learning_rate=cfg.training.learning_rate, 
        weight_decay=cfg.training.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_loss = float("inf")
    ckpt = None

    if cfg.training.resume_path and Path(cfg.training.resume_path).exists():
        ckpt = torch.load(cfg.training.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from: {cfg.training.resume_path} (epoch {start_epoch})")

    if num_gpus > 1:    
        model = nn.DataParallel(model)
        print(f"Using nn.DataParallel on {num_gpus} GPUs")


    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        train_loss, train_acc = train_one_epoch(
            cfg, model, train_loader, optimizer, criterion, device, epoch, scaler, writer
        )
        val_loss, val_acc, f1 = validate(
            cfg, model, val_loader, criterion, device, epoch, writer
        )
        scheduler.step()

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        print(
            f"Epoch {epoch + 1} Train: Loss={train_loss:.4f} | Acc={train_acc:.2f}%"
            f" — Val: Loss={val_loss:.4f} | Acc={val_acc:.2f}% | F1={f1:.4f}"
            f" — lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best val loss: {best_val_loss:.4f}")

        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model = cast(nn.Module, raw_model)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "f1": f1,
                "stage": 'person',
                "exp_dir": str(exp_dir),
            },
            is_best,
            rank=0,
        )

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()

