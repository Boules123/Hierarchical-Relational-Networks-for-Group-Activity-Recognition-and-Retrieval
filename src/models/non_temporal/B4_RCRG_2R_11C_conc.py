import torch 
import torch.nn as nn
import torch.nn.functional as F

from configs.read_yml_cfg import get_config
from training.training_group import train
from infere_model.inference import test
from models.relational_layer import RelationalLayer

class B4_RCRG_2R_11C_conc(nn.Module):
    def __init__(self, person_cls):
        super().__init__()
        self.person_cls = person_cls.backbone
        
        for param in self.person_cls.parameters():
            param.requires_grad = False

        self.relational_layer1 = RelationalLayer(2048, 256)
        self.relational_layer2 = RelationalLayer(256, 128)
        
        self.fc = nn.Sequential(
            nn.Linear(12*128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)
        )

    def configure_optimizers(self, learning_rate=1e-4, weight_decay=0.01):
        """
        for features: 
        1 - fused 
        2 - group param 
        3 - freeze param 
        """
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2:
                nodecay_params.append(param)

            else:
                decay_params.append(param)
                
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
                
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            fused=use_fused
        )
        
        return optimizer


if __name__ == "__main__":
    cfg = get_config("configs/B4_RCRG_2R_11C.yml")
    train(cfg)
    test(cfg)

#   Accuracy: 85.32%  |  F1: 0.8522
#               precision    recall  f1-score   support

#        r_set       0.74      0.89      0.81      1728
#      r_spike       0.96      0.83      0.89      1557
#       r-pass       0.91      0.76      0.83      1890
#   r_winpoint       0.98      0.86      0.92       783
#   l_winpoint       0.86      0.96      0.91       918
#       l-pass       0.77      0.96      0.86      2034
#      l-spike       0.86      0.92      0.89      1611
#        l_set       0.94      0.67      0.78      1512

#     accuracy                           0.85     12033
#    macro avg       0.88      0.86      0.86     12033
# weighted avg       0.87      0.85      0.85     12033
