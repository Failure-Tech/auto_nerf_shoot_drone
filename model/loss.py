import torch
import torch.nn as nn
import numpy as np

class ComputeLoss():
    def __init__(self, pred, target_xy, target_sin, target_cos, target_conf):
        super(ComputeLoss, self).__init__()
        self.xy_pred = pred[:, 0:2]
        self.sin_pred = pred[:, 2]
        self.cos_pred = pred[:, 3]
        self.pred_conf = pred[:, 4]

        self.target_xy = target_xy
        self.target_sin = target_sin
        self.target_cos = target_cos
        self.target_conf = target_conf

        self.xy_loss = nn.SmoothL1Loss()
        self.angle_loss = nn.MSELoss()
        self.conf_loss = nn.BCEWithLogitsLoss()

    def compute_loss(self) -> torch.Tensor:
        total_xy_loss = self.xy_loss(self.xy_pred, self.target_xy)
        total_angle_loss = self.angle_loss(self.sin_pred, self.target_sin) + self.angle_loss(self.cos_pred, self.target_cos)
        total_conf_loss = self.conf_loss(self.target_conf, self.pred_conf)
        

        error_prop = np.mean(np.pow((np.pow(self.sin_pred, 2) + np.pow(self.cos_pred, 2)-1), 2))
        # error_prop = ((sin_pred**2 + cos_pred**2 - 1.0)**2).mean()

        total = total_xy_loss + total_angle_loss + total_conf_loss + 0.1*error_prop

        total_xy_loss.backward()
        total_angle_loss.backward()
        total_conf_loss.backward()

        return total