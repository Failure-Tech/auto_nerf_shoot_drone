import os
import openml
import cv2

custom_cache_dir = "./mpii_human_pose_meta_album"
os.makedirs(custom_cache_dir, exist_ok=True)

openml.config.set_root_cache_directory(custom_cache_dir)

dataset = openml.datasets.get_dataset(44334, download_data=True, download_all_files=True)

# x, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format="images",
#     target=dataset.default_target_attribute
# )

# print(x.dtypes)

# cv2.imshow("test", x)
# cv2.waitKey(0)

# features_path = os.path.join(custom_cache_dir, "features.csv")
# targets_path = os.path.join(custom_cache_dir, "targets.csv")

# x.to_csv(features_path, index=False)
# y.to_csv(targets_path, index=False)

# print(f"Dataset downloaded to {custom_cache_dir}")