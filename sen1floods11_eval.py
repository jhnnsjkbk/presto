import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast, Callable

import geopandas
from cropharvest.config import LABELS_FILENAME
from cropharvest.engineer import TestInstance
from torch import nn
from torch.optim import SGD, Adam

from einops import rearrange, reduce, repeat
from pyproj import Transformer
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import csv
import os
import numpy as np
import rasterio
from sklearn.metrics import jaccard_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    recall_score, fbeta_score, precision_recall_fscore_support

import presto
from presto.presto import Presto

from presto.utils import (
    default_model_path,
    device
)

regression = False
multilabel = False
dim = 224
num_outputs = dim*dim
start_month = 1
num_timesteps = 1
batch_size = 64
name = "sen1floods11"

path_to_flood_images = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/S2GeodnHand6Bands/"
path_to_labels = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/LabelHand/"
train_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_train_data_S2_geodn.txt"
valid_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_valid_data_S2_geodn.txt"
test_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_test_data_S2_geodn.txt"

class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.data)


def processAndAugment(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)

    im1 = Image.fromarray(im[0])  # red
    im2 = Image.fromarray(im[1])  # green
    im3 = Image.fromarray(im[2])  # blue
    im4 = Image.fromarray(im[3])  # NIR narrow
    im5 = Image.fromarray(im[4])  # SWIR1
    im6 = Image.fromarray(im[5])  # SWIR2
    label = Image.fromarray(label.squeeze())
    i, j, h, w = transforms.RandomCrop.get_params(im1, (dim, dim))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    im3 = F.crop(im3, i, j, h, w)
    im4 = F.crop(im4, i, j, h, w)
    im5 = F.crop(im5, i, j, h, w)
    im6 = F.crop(im6, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        im3 = F.hflip(im3)
        im4 = F.hflip(im4)
        im5 = F.hflip(im5)
        im6 = F.hflip(im6)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        im3 = F.vflip(im3)
        im4 = F.vflip(im4)
        im5 = F.vflip(im5)
        im6 = F.vflip(im6)
        label = F.vflip(label)

    norm = transforms.Normalize([0.107582, 0.13471393, 0.12520133, 0.3236181, 0.2341743, 0.15878009],
                                [0.07145836, 0.06783548, 0.07323416, 0.09489725, 0.07938496, 0.07089546])

    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im3).squeeze(),
                        transforms.ToTensor()(im4).squeeze(),
                        transforms.ToTensor()(im5).squeeze(),
                        transforms.ToTensor()(im6).squeeze(),
                        ))]
    train_images = [norm(im) for im in ims]
    train_images = torch.stack(train_images).reshape(6, dim, dim)
    train_labels = transforms.ToTensor()(label).squeeze()

    month = torch.tensor([6] * train_images.shape[0]).long()
    train_images = rearrange(train_images, 't h w -> t (h w)')
    train_labels = rearrange(train_labels, 'h w -> (h w)')
    # TODO: Check labels

    return train_images, train_labels, month


def processTestIm(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.107582, 0.13471393, 0.12520133, 0.3236181, 0.2341743, 0.15878009],
                                [0.07145836, 0.06783548, 0.07323416, 0.09489725, 0.07938496, 0.07089546])
    
    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    im_c3 = Image.fromarray(im[2]).resize((512, 512))
    im_c4 = Image.fromarray(im[3]).resize((512, 512))
    im_c5 = Image.fromarray(im[4]).resize((512, 512))
    im_c6 = Image.fromarray(im[5]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [F.crop(im_c1, 0, 0, 224, 224), F.crop(im_c1, 0, 224, 224, 224),
              F.crop(im_c1, 224, 0, 224, 224), F.crop(im_c1, 224, 224, 224, 224)]
      
    im_c2s = [F.crop(im_c2, 0, 0, 224, 224), F.crop(im_c2, 0, 224, 224, 224),
              F.crop(im_c2, 224, 0, 224, 224), F.crop(im_c2, 224, 224, 224, 224)]
      
    im_c3s = [F.crop(im_c3, 0, 0, 224, 224), F.crop(im_c3, 0, 224, 224, 224),
              F.crop(im_c3, 224, 0, 224, 224), F.crop(im_c3, 224, 224, 224, 224)]
    
    im_c4s = [F.crop(im_c4, 0, 0, 224, 224), F.crop(im_c4, 0, 224, 224, 224),
              F.crop(im_c4, 224, 0, 224, 224), F.crop(im_c4, 224, 224, 224, 224)]
    
    im_c5s = [F.crop(im_c5, 0, 0, 224, 224), F.crop(im_c5, 0, 224, 224, 224),
              F.crop(im_c5, 224, 0, 224, 224), F.crop(im_c5, 224, 224, 224, 224)]
    
    im_c6s = [F.crop(im_c6, 0, 0, 224, 224), F.crop(im_c6, 0, 224, 224, 224),
              F.crop(im_c6, 224, 0, 224, 224), F.crop(im_c6, 224, 224, 224, 224)]

    labels = [F.crop(label, 0, 0, 224, 224), F.crop(label, 0, 224, 224, 224),
              F.crop(label, 224, 0, 224, 224), F.crop(label, 224, 224, 224, 224)]

    ims = [torch.stack((transforms.ToTensor()(a).squeeze(),
                        transforms.ToTensor()(b).squeeze(),
                        transforms.ToTensor()(c).squeeze(),
                        transforms.ToTensor()(d).squeeze(),
                        transforms.ToTensor()(e).squeeze(),
                        transforms.ToTensor()(f).squeeze()))
           for (a, b, c, d, e, f) in zip(im_c1s, im_c2s, im_c3s, im_c4s, im_c5s, im_c6s)]

    train_images = [norm(im) for im in ims]
    train_images = torch.stack(train_images).reshape(4*6, dim, dim)

    train_labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    train_labels = torch.stack(train_labels)

    month = torch.tensor([6] * train_images.shape[0]).long()
    train_images = rearrange(train_images, 'p h w -> p (h w)')
    train_labels = rearrange(train_labels, 'p h w -> p (h w)')

    """     
    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()
    """
    return train_images, train_labels, month

def metrics(y_true, y_pred) -> dict():
    """
    Returns metrics
    """
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_true = np.uint8(y_true)
    ignore_index = np.argwhere(y_true == 2)
    y_true = np.delete(y_true, ignore_index)
    y_pred = np.delete(y_pred, ignore_index)

    # metrics
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    bal_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    precision_weighted = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    iou_score = jaccard_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f0_5 = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    f0_1 = fbeta_score(y_true, y_pred, average='weighted', beta=0.1)
    f10 = fbeta_score(y_true, y_pred, average='weighted', beta=10)
    precision_per_class, recall_per_class, fscore_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred)
    iou_score_per_class = jaccard_score(y_true=y_true, y_pred=y_pred, average=None)

    return {"accuracy": accuracy,
            "bal_accuracy":  bal_accuracy, 
            "precision": precision, 
            "precision_weighted": precision_weighted, 
            "recall": recall, 
            "recall_weighted": recall_weighted, 
            "iou_score": iou_score, 
            "f1": f1, 
            "f1_micro": f1_micro, 
            "f1_macro": f1_macro, 
            "f0_5": f0_5, 
            "f0_1": f0_1, 
            "f10": f10, 
            "precision_per_class": precision_per_class, 
            "recall_per_class": recall_per_class, 
            "fscore_per_class": fscore_per_class, 
            "support_per_class": support_per_class, 
            "iou_score_per_class": iou_score_per_class
            }



def getArrFlood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(os.path.join("files/", im_fname)):
            continue
        arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
        arr_y = getArrFlood(os.path.join("files/", mask_fname))
        arr_y[arr_y == -1] = 0

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data


def load_flood_train_data(input_root, label_root):
    fname = train_split
    training_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            training_files.append(
                tuple((input_root + line[0] + "_S2GeodnHand.tif", label_root + line[0] + "_LabelHand.tif")))

    return download_flood_water_data_from_list(training_files)


def load_flood_valid_data(input_root, label_root):
    fname = valid_split
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            validation_files.append(
                tuple((input_root + line[0] + "_S2GeodnHand.tif", label_root + line[0] + "_LabelHand.tif")))

    return download_flood_water_data_from_list(validation_files)


def load_flood_test_data(input_root, label_root):
    fname = test_split
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(
                tuple((input_root + line[0] + "_S2GeodnHand.tif", label_root + line[0] + "_LabelHand.tif")))

    return download_flood_water_data_from_list(testing_files)


def finetune(pretrained_model, mask: Optional[np.ndarray] = None):
    print("finetune")
    print(pretrained_model)
    lr, num_grad_steps, k = 0.001, 250, 10
    model = construct_finetuning_model(pretrained_model)

    opt = SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="mean")
    batch_mask = mask_to_batch_tensor(mask, k)

    for i in range(1):
        if i != 0:
            model.train()
            opt.zero_grad()

        for (x, labels, month) in tqdm(dl):
            preds = model(
                x.to(device).float(),
                mask=None,
                dynamic_world=None,
                latlons=None,
                month=month,
            ).squeeze(dim=1)

        loss = loss_fn(preds, labels.squeeze().to(device).float())
        print("loss: ", loss)

        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(finetuned_model,
            pretrained_model=None,
            mask: Optional[np.ndarray] = None,
            ) -> Dict:
    finetuned_model.eval()
    """ 
    for (x, labels, month) in tqdm(dl):
        preds = model(
                x.to(device).float(),
                mask=None,
                dynamic_world=None,
                latlons=None,
                month=month,
            ).squeeze(dim=1)
    """
    for (x, labels, month) in tqdm(test_dl):
        preds = (
            finetuned_model(
                x.to(device).float(),
                mask=None,
                dynamic_world=None,
                latlons=None,
                month=month,
            )
                .squeeze(dim=1)
                .cpu()
                .numpy()
            )
        print(preds.shape)
        print(labels.detach().numpy().shape)
    results_dir = metrics(preds, labels.detach().numpy())

    return results_dir


def finetuning_results(
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
) -> Dict:
    for x in model_modes:
        assert x in ["finetune", "Regression", "Random Forest"]
    results_dict = {}
    if "finetune" in model_modes:
        model = finetune(pretrained_model, mask)
        results_dict.update(evaluate(model, None, mask))

    return results_dict


def construct_finetuning_model(pretrained_model):
    model = pretrained_model.construct_finetuning_model(
        num_outputs=num_outputs,
        regression=regression,
    )
    return model


def mask_to_batch_tensor(
        mask: Optional[np.ndarray], batch_size: int
) -> Optional[torch.Tensor]:
    if mask is not None:
        return repeat(torch.from_numpy(mask).to(device), "t c -> b t c", b=batch_size).float()
    return None


path_to_config = "config/default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))
model = Presto.construct(**model_kwargs)

# can't load state_dict because pretained embedding layer is smaller 
# model.load_state_dict(torch.load(default_model_path, map_location=device))
# model.to(device)

train_data = load_flood_train_data(path_to_flood_images, path_to_labels)
valid_data = load_flood_valid_data(path_to_flood_images, path_to_labels)
test_data = load_flood_test_data(path_to_flood_images, path_to_labels)

train_dataset = InMemoryDataset(data=train_data, 
                                transform=processAndAugment)

test_dataset = InMemoryDataset(data=test_data, 
                                transform=processTestIm)

dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)

test_dl = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

print(finetuning_results(pretrained_model=model,
                         model_modes=["finetune"]))

