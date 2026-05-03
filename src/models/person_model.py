import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PersonModel(nn.Module):
    def __init__(self):
        super(PersonModel, self).__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 9),
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
        x = self.backbone(x) # (b*n, 2048, 1, 1)
        x = x.view(b*n, -1) # (b*n, 2048)
        out = self.fc(x) # (b*n, 9)        
        return out
