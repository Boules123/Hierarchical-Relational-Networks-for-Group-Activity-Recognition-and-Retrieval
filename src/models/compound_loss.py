"""
For one stage model (person + scene)  
"""


import torch.nn as nn 

class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ce_loss_person = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_loss_scene = nn.CrossEntropyLoss()

    def forward(self, person_logits, scene_logits, person_labels, scene_labels):
        loss_person = self.ce_loss_person(person_logits, person_labels)
        loss_scene = self.ce_loss_scene(scene_logits, scene_labels)
        total_loss = self.alpha * loss_person + (1 - self.alpha) * loss_scene
        return total_loss 