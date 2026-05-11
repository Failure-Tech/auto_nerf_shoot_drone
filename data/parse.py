import json

with open("mpii_pose_256_256_yolo_annots.json", "r") as file:
    curr_read = json.load(file)

for value in curr_read:
    if (value["average_confidence"] == "unknown"):
        curr_read.remove(value)

with open("mpii_pose_256_256_parsed_conf.json", "w") as f:
    json.dump(curr_read, f, indent=2)