import torch 
import torch.nn as nn
import torch.nn.functional as F

from configs.read_yml_cfg import get_config
from training.training_group import train
from infere_model.inference import test

class B1_NoRelations(nn.Module):
    def __init__(self, person_cls, output_dim=128):
        super().__init__()
        self.person_cls = person_cls.backbone
        
        for param in self.person_cls.parameters():
            param.requires_grad = False
        
        self.shared_layer = nn.Linear(2048, output_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim, 8)
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
        x = x.view(b*n, -1)
        x = self.shared_layer(x) # (b*n, output_dim)
        
        x = x.view(b, n, -1) # (b, n, output_dim)
        left_team = x[:, :n//2, :].max(dim=1)[0] # (b, output_dim)
        right_team = x[:, n//2:, :].max(dim=1)[0] # (b, output_dim)
        
        combined = torch.cat([left_team, right_team], dim=1) # (b, output_dim*2)
        out = self.fc_layers(combined) # (b, 8)
        return out



if __name__ == "__main__":
    cfg = get_config("configs/B1_NoRelations.yml")
    train(cfg)
    test(cfg)


# Accuracy: 86.63%  |  F1: 0.8656

#               precision    recall  f1-score   support

#        r_set       0.81      0.86      0.83      1728
#      r_spike       0.84      0.92      0.88      1557
#       r-pass       0.87      0.76      0.81      1890
#   r_winpoint       0.95      0.89      0.92       783
#   l_winpoint       0.91      0.95      0.93       918
#       l-pass       0.86      0.88      0.87      2034
#      l-spike       0.85      0.93      0.89      1611
#        l_set       0.91      0.82      0.86      1512

#     accuracy                           0.87     12033
#    macro avg       0.88      0.87      0.87     12033
# weighted avg       0.87      0.87      0.87     12033
