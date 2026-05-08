import torch
import torch.nn as nn
import numpy as np

xy_loss = nn.SmoothL1Loss()
angle_loss = nn.MSELoss()
conf_loss = nn.BCEWithLogitsLoss()

class ComputeLoss():
    def __init__(self, pred, target_xy, target_sin, target_cos, target_conf):
        super(ComputeLoss, self).__init__()
        self.xy_pred = pred[:2]
        self.sin_pred = pred[2]
        self.cos_pred = pred[3]
        self.pred_conf = pred[4]

        self.target_xy = target_xy
        self.target_sin = target_sin
        self.target_cos = target_cos
        self.target_conf = target_conf

    def compute_loss(self) -> torch.tensor:
        total_xy_loss = xy_loss(self.xy_pred, self.target_xy)
        total_angle_loss = angle_loss(self.sin_pred, self.target_sin) + angle_loss(self.cos_pred, self.target_cos)
        total_conf_loss = conf_loss(self.target_conf, self.pred_conf)

        # error_prop = np.mean(np.pow((np.pow(self.sin_pred.detach().numpy(), 2) + np.pow(self.cos_pred.detach().numpy(), 2)-1), 2))
        # error_prop = ((sin_pred**2 + cos_pred**2 - 1.0)**2).mean()
        error_prop = ((self.sin_pred**2 + self.cos_pred**2)-1)**2
        error_prop = error_prop.mean()

        total = total_xy_loss + total_angle_loss + total_conf_loss + 0.1*error_prop
        
        total.backward()
        
        return total