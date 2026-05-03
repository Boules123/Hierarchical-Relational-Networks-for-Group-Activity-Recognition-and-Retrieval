import os
from pathlib import Path
import torch

def save_checkpoint(checkpoint, is_best=False, rank=0):
    """
    Save the best model and checkpoint model.
    """
    if rank != 0:
        return

    checkpoint_path = os.path.join(checkpoint['exp_dir'], f"checkpoint_epoch_{checkpoint['epoch']}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
 
    if is_best:
        best_model_path = os.path.join(checkpoint['exp_dir'], 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to {best_model_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None, use_ema=False):
    """
    Load model for resume training and test model 
    """
    if device is None:
        device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_key = "ema_state_dict" if use_ema and "ema_state_dict" in checkpoint else "model_state_dict"
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain required key: {state_key}")
    model.load_state_dict(checkpoint[state_key])

    if optimizer is None:
        return model
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    config = checkpoint.get("config")
    exp_dir = checkpoint.get("exp_dir")
    
    if "optimizer_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain optimizer_state_dict")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, config, exp_dir, start_epoch


def resolve_inference_checkpoint(resume_path, save_dir, baseline_name):
    """Resolve checkpoint path for inference."""
    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
        return str(ckpt_path)

    save_dir = Path(save_dir)
    candidates = sorted(
        save_dir.glob(f"{baseline_name}_*/best_model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for {baseline_name}. "
            f"Expected files like '{baseline_name}_*/best_model.pth' in {save_dir}"
        )

    return str(candidates[0])

def get_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, display=False):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    tick_labels = class_names if class_names is not None else "auto"
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    if display:
        plt.show()
    plt.close()

def get_classification_report(y_true, y_pred, class_names=None):
    from sklearn.metrics import classification_report
    if class_names is None:
        report = classification_report(y_true, y_pred)
    else:
        report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    return report