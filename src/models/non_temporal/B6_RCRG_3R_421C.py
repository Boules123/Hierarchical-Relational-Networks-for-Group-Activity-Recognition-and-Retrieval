import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from configs.read_yml_cfg import get_config
from training.training_group import train
from infere_model.inference import test
from models.relational_layer import RelationalLayer

class B6_RCRG_3R_421C(nn.Module):
    def __init__(self, person_cls):
        super().__init__()
        self.person_cls = person_cls.backbone
        
        for param in self.person_cls.parameters():
            param.requires_grad = False
        
        self.relational_layer1 = RelationalLayer(2048, 512)
        self.relational_layer2 = RelationalLayer(512, 256)
        self.relational_layer3 = RelationalLayer(256, 128)
        
        self.fc = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 8)
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
        x = x.view(b, n, -1) # (b, n, 2048)
        
        c1 = self.relational_layer1(x[:, 0:3, :]) # (b, 3, 512)
        c2 = self.relational_layer1(x[:, 3:6, :]) # (b, 3, 512)
        c3 = self.relational_layer1(x[:, 6:9, :]) # (b, 3, 512)
        c4 = self.relational_layer1(x[:, 9:12, :]) # (b, 3, 512)
        x = torch.cat([c1, c2, c3, c4], dim=1) # (b, n, 512)
        
        left = self.relational_layer2(x[:, :n//2, :]) # (b, 6, 256)
        right = self.relational_layer2(x[:, n//2:, :]) # (b, 6, 256)
        x = torch.cat([left, right], dim=1) # (b, n, 256)
        
        x = self.relational_layer3(x) # (b, n, 128)
        left_team = x[:, :n//2, :].max(dim=1)[0] # (b, 128)
        right_team = x[:, n//2:, :].max(dim=1)[0] # (b, 128)
        combined = torch.cat([left_team, right_team], dim=1) # (b, 256)
        
        out = self.fc(combined) # (b, 8)
        return out

if __name__ == "__main__":
    cfg = get_config("configs/B6_RCRG_3R_421C.yml")
    train(cfg)
    test(cfg)

#   Accuracy: 87.43%  |  F1: 0.8743

#               precision    recall  f1-score   support

#        r_set       0.85      0.83      0.84      1728
#      r_spike       0.90      0.91      0.90      1557
#       r-pass       0.84      0.86      0.85      1890
#   r_winpoint       0.95      0.81      0.87       783
#   l_winpoint       0.93      0.87      0.90       918
#       l-pass       0.85      0.92      0.88      2034
#      l-spike       0.90      0.91      0.90      1611
#        l_set       0.86      0.85      0.86      1512

#     accuracy                           0.87     12033
#    macro avg       0.88      0.87      0.88     12033
# weighted avg       0.88      0.87      0.87     12033
