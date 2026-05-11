import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json

path = "./mpii_human_pose_v1/images"
# path = "./mpii_human_pose_meta_album/org/openml/www/datasets/44334/ACT_410_Extended/images"
model = YOLO("yolo26x-pose.pt")

# USE THIS FOR FINAL JSON STRUCTURED OUTPUT (on each index)
keypoint_label = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

data = []

curr_total_index = 0
for root, _, filenames in os.walk(path):
        print(f"LEN: {len(filenames)}")
        for file in filenames:
            if (curr_total_index < len(filenames)):
                file_path = (root + "/" + file).replace("\\", "/")
                abs_file_path = os.path.abspath(file_path).replace("\\", "/")
                # print(f"FILE PATH: {abs_file_path}")

                curr_unresize_img = np.array(Image.open(abs_file_path))
                resized_image = cv2.resize(curr_unresize_img, (255, 255))

                # cv2.imshow("test", resized_image)
                # cv2.waitKey(0)

                # print(f"AFTER RESIZE SHAPE: {resized_image.shape}")
                # result_metrics = model.predict(abs_file_path)
                result_metrics = model.predict(resized_image)

                curr_metric_obj = {}
                boxes_metrics = {}
                keypoints_joint = {}

                curr_metric_obj["img_file_path"] = abs_file_path

                for result in result_metrics:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    masks = result.masks
                    probs = result.probs
                    obb = result.obb

                    # avg_conf = np.average([x.item() for x in keypoints.conf])
                    avg_conf = keypoints.conf.mean().item()
                    if (not np.isnan(avg_conf)):
                        print(f"AVG CONF: {avg_conf}")
                        curr_metric_obj["average_confidence"] = avg_conf
                    else:
                        curr_metric_obj["average_confidence"] = "unknown"

                    curr_metric_obj["array_confidence"] = keypoints.conf.flatten().tolist()

                    curr_img = cv2.imread(file_path)

                    for i in range(len(boxes.xywhn)):
                        x1, y1, w, h = boxes.xywhn[i]
                        boxes_metrics[f"center_x_y_point_{i+1}"] = [x1.item(), y1.item()]
                        # curr_img[int(y1-15):int(y1), int(x1-15):int(x1)] = [0, 0, 255]

                    curr_metric_obj["center_points"] = boxes_metrics

                    # cv2.imshow("test", curr_img)
                    # cv2.waitKey(0)

                    # print(f"SHAPE: {keypoints.shape}")
                    # result.show()

                    for pixel_tensors in keypoints.xy:
                        for i in range(len(pixel_tensors)):
                            point_x, point_y = pixel_tensors[i]
                            # curr_img[int(point_y.item()-10):int(point_y.item()), int(point_x.item()-50):int(point_x.item())] = [0, 0, 255]
                            curr_keypoint_label = keypoint_label[i]
                            keypoints_joint[curr_keypoint_label] = [point_x.item(), point_y.item()]
                            # print(f"Current Keypoint Label: {keypoints_joint[curr_keypoint_label]}")
                    
                    curr_metric_obj["keypoints"] = keypoints_joint

                    # print(curr_metric_obj)

                    data.append(curr_metric_obj)
                    curr_total_index+=1
            else:
                 break
        break 
        # cv2.imshow("test", curr_img)
        # cv2.waitKey(0)

        # print(f"KEYPOINTS: {keypoints.xy}, AVG CONFIDENCE: {avg_conf:.4f}")
    #     break
    # break

for value in data:
    if (value["average_confidence"] == "unknown"):
        data.remove(value)
    else:
        continue

with open("mpii_pose_256_256_yolo_annots.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved json file")