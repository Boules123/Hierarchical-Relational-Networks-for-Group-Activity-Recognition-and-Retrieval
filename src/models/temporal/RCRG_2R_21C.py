import torch 
import torch.nn as nn

from relational_layer import RelationalLayer
from training.training_group import train 
from infere_model.inference import test 
from configs.read_yml_cfg import get_config

class RCRG_2R_21C(nn.Module):
    def __init__(self, person_cls):
        super().__init__()
        self.person_model = person_cls.backbone
        
        for param in self.person_model.parameters():
            param.requires_grad = False
        
        self.rl1 = RelationalLayer(512, 256)
        self.rl2 = RelationalLayer(256, 128)
        
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=512, 
            batch_first=True, 
            num_layers=1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8)
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
        b, n, t, c, h, w = x.size()
        
        # Using reshape instead of view for safety
        x = x.reshape(b*n*t, c, h, w)
        x = self.person_model(x) 
        
        x = x.reshape(b*n, t, -1)
        out, _ = self.lstm(x) 
        
        # Slicing creates non-contiguous tensors, reshape handles it safely
        x = out[:, -1, :].reshape(b, n, -1) 
        
        left = self.rl1(x[:, :n//2, :]) 
        right = self.rl1(x[:, n//2:, :]) 
        
        x1 = torch.cat([left, right], dim=1) 
        x2 = self.rl2(x1) 
        
        x = x2.max(dim=1)[0] 
        
        logits = self.fc(x) 
        return logits

    
if __name__ == "__main__":
    cfg = get_config("configs/RCRG_2R_21C.yml")
    train(cfg, seq=True)
    test(cfg, seq=True)


# Not Run it yet