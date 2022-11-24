import os
import sys
import shutil
import argparse

import torch

from models import vgg
from data_loader import get_test_dataloader

import pickle
with open("idx_to_class.pkl", 'rb') as f :
    idx_to_class = pickle.load(f)


def classify(image_path) :
    # move the image to 
    image_name = image_path.split("\\")[-1].split("/")[-1]
    if not os.path.exists("input/input_image") : os.makedirs("input/input_image")
    shutil.copy(image_path, f'input/input_image/{image_name}')
    loader = get_test_dataloader(path='input')[0]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg()
    model.load_state_dict(torch.load('models/vgg.pt', map_location=torch.device('cpu')))
    model.eval()

    for image, _ in loader :
        pass
    shutil.rmtree('input')
    
    output = model(image.to(device))
    idx = torch.argmax(output[0]).item()
    return f"The vechile is : {idx_to_class[idx]}"

    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Add path of vechile image')
    
    parser.add_argument('--img',
                         metavar='image',
                         type=str,
                         help='the path to jpg')
    
    args = parser.parse_args()    
    image_path = args.img
    
    if not os.path.isfile(image_path):
        print('The path specified does not exist')
        sys.exit()


    print(classify(image_path))