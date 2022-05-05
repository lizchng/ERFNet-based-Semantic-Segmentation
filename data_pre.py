import os
from PIL import Image
import re
import numpy as np

def pre():
    for folder in os.listdir('./dataset/json'):
        for file in os.listdir(os.path.join('./dataset/json', folder)):
            polygons = open(f'./dataset/Zhoushan/gtFine/{folder}_{file[:-5].replace(".", "_")}_gtFine_polygons.json', 'w')
            f = open(os.path.join('./dataset/json', folder, file), 'r').readlines()
            polygons.writelines(f)

    for folder in os.listdir('./dataset/image'):
        for file in os.listdir(os.path.join('./dataset/image', folder)):
            img = Image.open(os.path.join('./dataset/image', folder, file))
            img.save(f'./dataset/Zhoushan/leftImg8bit/{folder}_{file[:-4].replace(".", "_")}_leftImg8bit.png')


def pos():
    base_path = './dataset/dataset'
    folders = os.listdir(base_path)
    folders = sorted(folders)
    for folder in folders:
        for file in os.listdir(os.path.join(base_path, folder)):
            img_path = os.path.join(base_path, folder, file, 'LeopardCamera1')
            for itm in os.listdir(img_path):
                if 'png' in itm:
                    path = os.path.join(img_path, itm)
                    print(path)
                    seg_data(path)

def seg_data(img_path):
    if not os.path.exists('./dataset/unlabeled/'):
        os.mkdir('./dataset/unlabeled/')
    img_name = img_path.split('/')[-1]
    match = re.search(r'group\d\d\d\d', img_path)
    group = match.group()
    new_name = f"{group}_{img_name[:-4].replace('.', '_')}_leftImg8bit.png"
    assert match, ''
    if not os.path.exists(f'./dataset/unlabeled/{group}/leftImg8bit/'):
        os.mkdir(f'./dataset/unlabeled/{group}/leftImg8bit/')
    image = np.zeros((704, 2000, 3), dtype=np.uint8)
    img = Image.open(img_path)
    image = np.asarray(img)[500:1204, 750:2750,:]
    Image.fromarray(image).save(f'./dataset/unlabeled/{group}/{new_name}')

pre()
pos()