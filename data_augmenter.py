import os
import cv2
import shutil
from tqdm import tqdm
import imgaug.augmenters as iaa


def create_augmented_data():
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                        iaa.Crop(percent=(0, 0.1)),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                        iaa.LinearContrast((0.2, 2)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        iaa.Multiply((0.8, 1.2), per_channel=0.2),
                        iaa.Affine(scale={"x": (1, 1.3), "y": (1, 1.3)},
                                    rotate=(-7, 7),
                                    fit_output=True,
                                    mode='edge'),])

    if not os.path.exists(f"data/train/augmented_data") : os.mkdir("data/train/augmented_data")
    print("Start augmenting data :\n")
    for dir in tqdm(os.listdir("data/train/original_data")):
        idx = 0
        if not os.path.exists(f"data/train/augmented_data/{dir}") : shutil.copytree(f"data/train/original_data/{dir}", f"data/train/augmented_data/{dir}")
        if len(os.listdir(f"data/train/augmented_data/{dir}")) < 1000 : 
            class_images = [cv2.cvtColor(cv2.imread(f"data/train/original_data/{dir}/{img}"), cv2.COLOR_BGR2RGB) for img in os.listdir(f"data/train/original_data/{dir}")]
            while len(os.listdir(f"data/train/augmented_data/{dir}")) < 1000:
                augs = seq.augment_images(class_images)
                for img in augs :
                    cv2.imwrite(f"data/train/augmented_data/{dir}/augmented_{idx}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    idx += 1



if __name__ == "__main__" :
        create_augmented_data()