import torch 
import torch.nn as nn
import torch.nn.functional as F

from configs.read_yml_cfg import get_config
from training.training_group import train
from infere_model.inference import test
from models.relational_layer import RelationalLayer

class B5_RCRG_2R_21C_conc(nn.Module):
    def __init__(self, person_cls):
        super().__init__()
        self.person_cls = person_cls.backbone
        
        for param in self.person_cls.parameters():
            param.requires_grad = False
        
        self.relational_layer1 = RelationalLayer(2048, 256)
        self.relational_layer2 = RelationalLayer(256, 128)
        
        self.fc = nn.Sequential(
            nn.Linear(128*12, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 8)
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
    
    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(b*n, c, h, w)
        x = self.person_cls(x) # (b*n, 2048, 1, 1)
        x = x.view(b, n, -1)
        
        left = self.relational_layer1(x[:, :n//2, :]) # (b, n//2, 256)
        right = self.relational_layer1(x[:, n//2:, :]) # (b, n//2, 256)
        x1 = torch.cat([left, right], dim=1) # (b, n, 256)
        
        x2 = self.relational_layer2(x1) # (b, n, 128)
        
        x = x2.view(b, -1) # (b, n*128)

        out = self.fc(x) # (b, 8)
        
        
        return out


if __name__ == "__main__":
    cfg = get_config("configs/B5_RCRG_2R_21C.yml")
    train(cfg)
    test(cfg)

#   Accuracy: 83.81%  |  F1: 0.8363
#               precision    recall  f1-score   support

#        r_set       0.72      0.87      0.79      1728
#      r_spike       0.82      0.91      0.86      1557
#       r-pass       0.92      0.63      0.75      1890
#   r_winpoint       0.95      0.88      0.91       783
#   l_winpoint       0.90      0.92      0.91       918
#       l-pass       0.88      0.83      0.85      2034
#      l-spike       0.82      0.93      0.87      1611
#        l_set       0.85      0.83      0.84      1512

#     accuracy                           0.84     12033
#    macro avg       0.86      0.85      0.85     12033
# weighted avg       0.85      0.84      0.84     12033
