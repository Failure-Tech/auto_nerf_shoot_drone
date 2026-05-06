from torchvision.io import decode_image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

data_path = "./data/data.json"
df = pd.read_json(data_path)

class MPII_Human_Pose_YOLO(Dataset):
    def __init__(self, data_annotations_path: str, transform=None, target_transform=None):
        super(MPII_Human_Pose_YOLO, self).__init__()
        self.annots = pd.read_json(data_annotations_path)

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        img_path = self.annots[idx]["img_file_path"]
        image = decode_image(img_path)

        # form vector for shoulder-hip midpoints avg[left_shoulder, right_shoulder] * avg[left_hip, right_hip] calculate angle w/ sin & cos
        left_shoulder = self.annots[idx]["left_shoulder"]
        right_shoulder = self.annots[idx]["right_shoulder"]

        # y, x
        shoulder_midpoint = [np.sqrt(np.square(left_shoulder[0]-right_shoulder[0])), np.sqrt(np.square(left_shoulder[1] - right_shoulder[1]))]

        left_hip = self.annots[idx]["left_hip"]
        right_hip = self.annots[idx]["right_hip"]
        hip_midpoint = [np.sqrt(np.square(left_hip[0]-right_hip[0])), np.sqrt(np.square(left_hip[1] - right_hip[1]))]

        # x, y
        point_vector = [np.abs(shoulder_midpoint[1]-hip_midpoint[1]), np.abs(shoulder_midpoint[0]-hip_midpoint[0])]

        theta = np.atan2(point_vector[1], point_vector[0])
        sin_label = np.sin(theta)
        cos_label = np.cos(theta)

        conf_label = self.annots[idx]["average_confidence"]

        # all center_point vals detected
        center_points_xy = []

        # all xy labels (all detected)
        for _, value in self.annots[idx]["center_points"].items():
            center_points_xy.append(value)

        labels = [center_points_xy, sin_label, cos_label, conf_label]

        return image, labels