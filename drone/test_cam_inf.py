import sys
sys.path.insert(0, "../model")

import torch
from poseRegressor import PoseRegressorModel
from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import transforms
import math

def load_model(curr_statedict_path: str):
    model = PoseRegressorModel(in_channels=3, hidden_channels=32)

    # "model_state_dict"
    curr_json_obj = torch.load(curr_statedict_path)
    print(f"dtype: {type(curr_json_obj)}")

    model.load_state_dict(curr_json_obj["model_state_dict"])
    model.eval()

    return model

def preprocess_img(img_path: str) -> torch.tensor:
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    transformed_image = transform(img)
    transformed_image = transformed_image.unsqueeze(0)

    # img = img.resize((256, 256))

    # img = np.array(img).astype(np.float32)

    # cv2.imshow("test", img)
    # cv2.waitKey(0)

    # torch.tensor(img)

    return transformed_image


# curr_statedict_path = "../model/checkpoints/checkpoint_40.pt"
curr_statedict_path = "../model/checkpoints/checkpoint_30.pt"

model = load_model(curr_statedict_path=curr_statedict_path)

# 46659
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

img_path = "./construction_worker_test.png"
input_image = preprocess_img(img_path=img_path)
target_x, target_y, target_sin, target_cos, target_conf = model(input_image).squeeze(0).to(torch.float32)

xy_coords = 255*np.array([target_x.item(), target_y.item()])
print(f"COORDS: {xy_coords}")

theta1 = np.asin(target_sin.item())*180/math.pi
theta2 = np.acos(target_cos.item())

conf = target_conf.item()*100

print(f"THETAS: {theta1}, {theta2}")

print(f"CONF: {conf}%")

test_cv = cv2.imread(img_path)
test_cv = cv2.resize(test_cv, (256, 256))

# draw black pixels to screen to visualize
test_cv[int(xy_coords[1]-10):int(xy_coords[1]+10), int(xy_coords[0]-10):int(xy_coords[0]+10)] = [0, 0, 0]

cv2.imshow("test_cv", test_cv)
cv2.waitKey(0)

# output_arr = [target_x, target_y, target_sin, target_cos, target_conf]

# print(output_arr)

