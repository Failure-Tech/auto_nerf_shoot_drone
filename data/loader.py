import torch
from torchvision.io import decode_image
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset
import json
import numpy as np

class MPII_Human_Pose_YOLO(Dataset):
    def __init__(self, data_annotations_path: str):
        super(MPII_Human_Pose_YOLO, self).__init__()
        with open(data_annotations_path, "r") as file:
            self.annots = json.load(file)

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        img_path = self.annots[idx]["img_file_path"]

        # image = decode_image(img_path)
        image = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        # image_tensor = transform(image).cpu().numpy()
        # if image_tensor.ndim==3:
        #     image_tensor = np.transpose(image_tensor, (1,2,0))

        # image_tensor = Image.fromarray(image_tensor.astype())

        # form vector for shoulder-hip midpoints avg[left_shoulder, right_shoulder] * avg[left_hip, right_hip] 
        # calculate angle w/ sin & cos or sm shi
        
        left_shoulder = self.annots[idx]["keypoints"]["left_shoulder"]
        right_shoulder = self.annots[idx]["keypoints"]["right_shoulder"]

        # y, x
        # shoulder_midpoint = [np.sqrt(np.square(left_shoulder[0]-right_shoulder[0])), np.sqrt(np.square(left_shoulder[1] - right_shoulder[1]))]
        shoulder_midpoint = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]


        left_hip = self.annots[idx]["keypoints"]["left_hip"]
        right_hip = self.annots[idx]["keypoints"]["right_hip"]

        # print(f"INDEX: {idx} RIGHT_HIP: {right_hip}")
        # hip_midpoint = [np.sqrt(np.square(left_hip[0]-right_hip[0])), np.sqrt(np.square(left_hip[1] - right_hip[1]))]
        hip_midpoint = [(right_hip[0]+left_hip[0])/2, (left_hip[1]+right_hip[1])/2]

        # x, y
        point_vector = [shoulder_midpoint[1]-hip_midpoint[1], shoulder_midpoint[0]-hip_midpoint[0]]

        theta = np.atan2(point_vector[1], point_vector[0])
        sin_label = np.sin(theta)
        cos_label = np.cos(theta)

        conf_label = self.annots[idx]["average_confidence"]

        # all center_point vals detected
        center_points_xy = []

        # center_counter = 19
        for _, value in self.annots[idx]["center_points"].items():
            # center_counter += 1
            center_points_xy.append(value)

        center_points_xy = np.array(center_points_xy)

        # add extra padding so i can avoid collate
        # sin_label_pad = np.full(shape=(center_counter, 2), fill_value=sin_label)
        # cos_label_pad = np.full(shape=(center_counter, 2), fill_value=cos_label)
        # conf_label_pad = np.full(shape=(center_counter, 2), fill_value=conf_label)

        # print(f"SHAPES\nSIN: {sin_label_pad.shape}\nCOS: {cos_label_pad.shape}\nCONF: {conf_label_pad.shape}\nCENTER: {center_points_xy.shape}")

        # targets = [center_points_xy, sin_label_pad, cos_label_pad, conf_label_pad]
        targets = [center_points_xy[0][1], center_points_xy[0][1], sin_label, cos_label, conf_label]
        targets = np.array(targets, dtype=np.float32)

        return image_tensor, targets
        