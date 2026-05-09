import sys
sys.path.insert(0, "../data")

import torch
from torch.utils.data import DataLoader
import loader
from poseRegressor import PoseRegressorModel
from loss import ComputeLoss
import matplotlib.pyplot as plt

def train_loop():
    # train_path = "../data/mpii_human_pose_yolo_annots.json"
    # train_path = "../data/mpii_meta_albumhuman_pose_yolo_annots.json"
    train_path = "../data/mpii_pose_256_refined.json"

    train_dataset = loader.MPII_Human_Pose_YOLO(train_path)

    # for i in range(10):
    #     img, label = train_dataset.__getitem__(i)
    #     # print(label[1])
    #     print("shpae: " + str(np.array(label).shape))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    model = PoseRegressorModel(in_channels=3, hidden_channels=32)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    num_epochs = 10

    losses = []

    for epoch in range(num_epochs):
        model.train()
        for img, targets in train_dataloader:
            # cv2.imshow("train_test", np.array(img))
            # cv2.waitKey(0)
            
            optimizer.zero_grad()
            preds = model(img)

            loss = ComputeLoss(preds, targets[0], targets[1], targets[2], targets[3])
            curr_loss = loss.compute_loss().item()

            optimizer.step()

        losses.append(curr_loss)
        print(f"Epoch: {epoch+1}, Loss: {curr_loss}")

    plt.plot([epoch for epoch in range(num_epochs)], losses)
    plt.savefig("train_loop_test_2.png", bbox_inches="tight")
    plt.show()
        
if __name__ == "__main__":
    train_loop()