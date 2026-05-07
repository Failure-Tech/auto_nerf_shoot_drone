import sys
sys.path.insert(0, "../data")

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import DataLoader
import loader
from poseRegressor import PoseRegressorModel
from loss import ComputeLoss
import cv2

def train_loop():
    # train_path = "../data/mpii_human_pose_yolo_annots.json"
    train_path = "../data/mpii_meta_albumhuman_pose_yolo_annots.json"

    train_dataset = loader.MPII_Human_Pose_YOLO(train_path, transform=None, target_transform=None)

    # for i in range(10):
    #     img, label = train_dataset.__getitem__(i)
    #     print(torch.tensor(label).shape)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    model = PoseRegressorModel(in_channels=3, hidden_channels=32)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        for img, targets in train_dataloader:
            cv2.imshow("train_test", np.array(img))
            cv2.waitKey(0)
            
            optimizer.zero_grad()
            preds = model(img)

            loss = ComputeLoss(preds, targets[0], targets[1], targets[2], targets[3])
            
            optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    train_loop()