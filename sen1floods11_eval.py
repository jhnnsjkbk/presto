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

import presto
from presto.presto import Presto

regression = False
multilabel = False
num_outputs = 1
start_month = 1
num_timesteps = 1

path_to_flood_images = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/S2GeodnHand6Bands/"
path_to_labels = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/LabelHand/"
train_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_train_data_S2_geodn.txt"
valid_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_valid_data_S2_geodn.txt"
test_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_test_data_S2_geodn.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def processAndAugment(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)

    im1 = Image.fromarray(im[0])  # red
    im2 = Image.fromarray(im[1])  # green
    im3 = Image.fromarray(im[2])  # blue
    im4 = Image.fromarray(im[3])  # NIR narrow
    label = Image.fromarray(label.squeeze())
    dim = 128
    i, j, h, w = transforms.RandomCrop.get_params(im1, (dim, dim))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    im3 = F.crop(im3, i, j, h, w)
    im4 = F.crop(im4, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        im3 = F.hflip(im3)
        im4 = F.hflip(im4)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        im3 = F.vflip(im3)
        im4 = F.vflip(im4)
        label = F.vflip(label)

    norm = transforms.Normalize([0.21531178, 0.20978154, 0.18528642, 0.48253757],
                                [0.10392396, 0.10210076, 0.11696766, 0.19680527])

    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im3).squeeze(),
                        transforms.ToTensor()(im4).squeeze()))]
    ims = [norm(im) for im in ims]
    im = torch.stack(ims).reshape(4, 1, dim, dim)
    label = transforms.ToTensor()(label).squeeze()
    # TODO: Check labels

    return im, label


def processTestIm(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [F.crop(im_c1, 0, 0, 128, 128), F.crop(im_c1, 0, 128, 128, 128),
              F.crop(im_c1, 128, 0, 128, 128), F.crop(im_c1, 128, 128, 128, 128)]
    im_c2s = [F.crop(im_c2, 0, 0, 128, 128), F.crop(im_c2, 0, 128, 128, 128),
              F.crop(im_c2, 128, 0, 128, 128), F.crop(im_c2, 128, 128, 128, 128)]
    labels = [F.crop(label, 0, 0, 128, 128), F.crop(label, 0, 128, 128, 128),
              F.crop(label, 128, 0, 128, 128), F.crop(label, 128, 128, 128, 128)]

    ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
                        transforms.ToTensor()(y).squeeze()))
           for (x, y) in zip(im_c1s, im_c2s)]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels


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
        arr_y[arr_y == -1] = 255

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


train_data = load_flood_train_data(path_to_flood_images, path_to_labels)
train_dataset = InMemoryDataset(train_data, processAndAugment)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, sampler=None,
                                           batch_sampler=None, num_workers=0, collate_fn=None,
                                           pin_memory=True, drop_last=False, timeout=0,
                                           worker_init_fn=None)
train_iter = iter(train_loader)

valid_data = load_flood_valid_data(path_to_flood_images, path_to_labels)
valid_dataset = InMemoryDataset(valid_data, processTestIm)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, sampler=None,
                                           batch_sampler=None, num_workers=0, collate_fn=lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                           pin_memory=True, drop_last=False, timeout=0,
                                           worker_init_fn=None)
valid_iter = iter(valid_loader)


@staticmethod
def dynamic_world_tifs_to_npy():
    def process_filename(filestem: str) -> Tuple[int, str]:
        r"""
        Given an exported sentinel file, process it to get the dataset
        it came from, and the index of that dataset
        """
        parts = filestem.split("_")[0].split("-")
        index = parts[0]
        dataset = "-".join(parts[1:])
        return int(index), dataset

    input_folder = cropharvest_data_dir() / DynamicWorldExporter.output_folder_name
    output_folder = cropharvest_data_dir() / "features/dynamic_world_arrays"
    labels = geopandas.read_file(cropharvest_data_dir() / LABELS_FILENAME)

    for filepath in tqdm(list(input_folder.glob("*.tif"))):
        index, dataset = process_filename(filepath.stem)
        output_filename = f"{index}_{dataset}.npy"
        if not (output_folder / output_filename).exists():
            rows = labels[((labels["dataset"] == dataset) & (labels["index"] == index))]
            row = rows.iloc[0]
            array, _, _ = DynamicWorldExporter.tif_to_npy(
                filepath, row["lat"], row["lon"], DEFAULT_NUM_TIMESTEPS
            )
            assert len(array) == DEFAULT_NUM_TIMESTEPS
            np.save(output_folder / output_filename, array)


def truncate_timesteps(x):
    if (num_timesteps is None) or (x is None):
        return x
    else:
        return x[:, : num_timesteps]


def finetune(pretrained_model, mask: Optional[np.ndarray] = None):
    print("finetune")
    print(pretrained_model)
    lr, num_grad_steps, k = 0.001, 250, 10
    model = construct_finetuning_model(pretrained_model)

    opt = SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="mean")
    batch_mask = mask_to_batch_tensor(mask, k)

    for i in range(num_grad_steps):
        if i != 0:
            model.train()
            opt.zero_grad()

    for i, data in enumerate(train_loader):
        # TODO: Sample or instead iter over dataloader?
        train_x, train_y = data[0], data[i]
        preds = model(
            truncate_timesteps(train_x.to(device).float()),
            mask=truncate_timesteps(batch_mask),
            dynamic_world=None,
            latlons=None,
            month=start_month,
        ).squeeze(dim=1)
        loss = loss_fn(preds, torch.from_numpy(train_y).to(device).float())

        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(
        finetuned_model,
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
) -> Dict:
    with tempfile.TemporaryDirectory() as results_dir:
        for test_id, test_instance, test_dw_instance in dataset.test_data(max_size=10000):
            savepath = Path(results_dir) / f"{test_id}.nc"

            test_x = truncate_timesteps(
                torch.from_numpy(S1_S2_ERA5_SRTM.normalize(test_instance.x)).to(device).float()
            )
            # mypy fails with these lines uncommented, but this is how we will
            # pass the other values to the model
            test_latlons_np = np.stack([test_instance.lats, test_instance.lons], axis=-1)
            test_latlon = torch.from_numpy(test_latlons_np).to(device).float()
            test_dw = truncate_timesteps(
                torch.from_numpy(test_dw_instance.x).to(device).long()
            )
            batch_mask = truncate_timesteps(
                mask_to_batch_tensor(mask, test_x.shape[0])
            )

            if isinstance(finetuned_model, FineTuningModel):
                finetuned_model.eval()
                preds = (
                    finetuned_model(
                        test_x,
                        dynamic_world=test_dw,
                        mask=batch_mask,
                        latlons=test_latlon,
                        month=start_month,
                    )
                        .squeeze(dim=1)
                        .cpu()
                        .numpy()
                )
            else:
                cast(Seq2Seq, pretrained_model).eval()
                encodings = (
                    cast(Seq2Seq, pretrained_model)
                        .encoder(
                        test_x,
                        dynamic_world=test_dw,
                        mask=batch_mask,
                        latlons=test_latlon,
                        month=start_month,
                    )
                        .cpu()
                        .numpy()
                )
                preds = finetuned_model.predict_proba(encodings)[:, 1]
            ds = test_instance.to_xarray(preds)
            ds.to_netcdf(savepath)

        all_nc_files = list(Path(results_dir).glob("*.nc"))
        combined_instance, combined_preds = TestInstance.load_from_nc(all_nc_files)
        combined_results = combined_instance.evaluate_predictions(combined_preds)

    prefix = finetuned_model.__class__.__name__
    return {f"{name}: {prefix}_{key}": val for key, val in combined_results.items()}


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

print(finetuning_results(pretrained_model=model,
                         model_modes=["finetune"]))
