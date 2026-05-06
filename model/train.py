import torch
from torch.utils.data import DataLoader
from data.loader import MPII_Human_Pose_YOLO
from poseRegressor import PoseRegressorModel
from loss import ComputeLoss

train_dataset = MPII_Human_Pose_YOLO("../data/mpii_human_pose_yolo_annots.json")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

model = PoseRegressorModel()

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for data, targets in train_dataloader:
        optimizer.zero_grad()
        preds = model(data)

        loss = ComputeLoss(preds, targets[0], targets[1], targets[2], targets[3])
        
        optimizer.step()
