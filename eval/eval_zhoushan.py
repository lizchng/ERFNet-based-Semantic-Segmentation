# Code to produce colored segmentation output in Pytorch for all cityscapes subsets
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib
import re

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 9

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((704, 2000), Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((704, 2000), Image.NEAREST),
    ToLabel(),
])

cityscapes_trainIds2labelIds = Compose([
    Relabel(8, 8),
    Relabel(7, 7),
    Relabel(6, 6),
    Relabel(5, 5),
    Relabel(4, 4),
    Relabel(3, 3),
    Relabel(2, 2),
    Relabel(1, 1),
    Relabel(0, 0),
    ToPILImage(),
])


def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # Import ERFNet model from the folder
    # Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    model = ERFNet(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    # model.load_state_dict(torch.load(args.state))
    # model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, groupId=''),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    for step, (images, filename) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            # labels = labels.cuda()

        inputs = Variable(images)
        # targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        label_id = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize()(label.unsqueeze(0))

        match = re.search(r'group\d\d\d\d', args.datadir)
        group = match.group()

        filenameSave = f"../dataset/predict_out/color/{group}/" + filename[0].split("leftImg8bit/")[1]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        # image_transform(label.byte()).save(filenameSave)
        label_save = ToPILImage()(label_color)
        label_save.save(filenameSave)

        filenameIdSave = f"../dataset/predict_out/gray/{group}/" + filename[0].split("leftImg8bit/")[1]
        os.makedirs(os.path.dirname(filenameIdSave), exist_ok=True)
        # image_transform(label.byte()).save(filenameSave)
        # labelId_save = ToPILImage()(label_id)
        label_id.save(filenameIdSave)

        if (args.visualize):
            vis.image(label_color.numpy())
        print(step, filenameSave)
        print(step, filenameIdSave)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../save/group0003/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    # parser.add_argument('--subset', default="")  # can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="../dataset/unlabeled/group0003")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
