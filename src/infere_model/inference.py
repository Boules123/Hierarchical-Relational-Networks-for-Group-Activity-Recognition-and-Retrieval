"""
Unified inference engine for all baselines.
Handles model evaluation, confusion matrix, and classification report.
"""

import os
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from sklearn.metrics import f1_score
import sys
from pathlib import Path
from torch.utils.data import DataLoader


import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data.dataset_loader import ACTIVITIES_LABELS, GroupActivityDataset
from utils.helpers import get_confusion_matrix, get_classification_report, load_checkpoint, resolve_inference_checkpoint
from models.model_registery import get_model
from models.person_model import PersonModel



def run_inference(model, loader, device, class_names=None, output_path=None):
    """
    Run inference and print metrics.
    "person" → evaluate person-level accuracy & F1
    "group"  → evaluate group-level accuracy & F1 (default)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            crops = batch[0].to(device)
            group_labels = batch[1].to(device)
            

            with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(crops)

            preds = outputs.argmax(1)
            labels = group_labels

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = (all_preds == all_labels).mean() * 100.0
    print(f"\n{'='*50}")
    print(f"  Accuracy: {acc:.2f}%  |  F1: {f1:.4f}")
    print(f"{'='*50}\n")

    if class_names and output_path:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"confusion_matrix.png")
        get_confusion_matrix(all_labels, all_preds, class_names=class_names, save_path=save_path, display=True)
        print(f"Confusion matrix saved to: {save_path}")
        get_classification_report(all_labels, all_preds, class_names=class_names)

    return acc, f1



def test(cfg, seq=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_ds = GroupActivityDataset(
        cfg.dataset.root_dir, 
        cfg.dataset.annot_dir, cfg.dataset.test_split,
        seq=seq,
        transform=transform,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size * max(num_gpus, 1),
        shuffle=False, num_workers=2,
    )

    ckpt_person = torch.load("load_your_person_checkpoint.pth", weights_only=False)
    person_cls = PersonModel().to(device)
    person_cls.load_state_dict(ckpt_person["model_state_dict"])

    model = get_model(cfg.experiment.exp_name, person_cls=person_cls).to(device)

    ckpt_path = "load_model_group_checkpoint.pth"
    loaded = load_checkpoint(ckpt_path, model, optimizer=None, device=device, use_ema=True)
    model = loaded[0] if isinstance(loaded, tuple) else loaded

    if num_gpus > 1:
        model = nn.DataParallel(model)

    print(f"Loaded checkpoint from: {ckpt_path}")
    run_inference(
        model, test_loader, device,
        class_names=list(ACTIVITIES_LABELS['group'].keys()),
        output_path=cfg.experiment.save_dir,
    )
