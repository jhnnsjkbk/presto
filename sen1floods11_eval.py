import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import geopandas
from cropharvest.config import LABELS_FILENAME
from cropharvest.engineer import TestInstance
from torch import nn
from torch.optim import SGD, Adam

from ..dataops import S1_S2_ERA5_SRTM, TAR_BUCKET
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .cropharvest_extensions import (
    DEFAULT_NUM_TIMESTEPS,
    CropHarvest,
    CropHarvestLabels,
    DynamicWorldExporter,
    Engineer,
    MultiClassCropHarvest,
    cropharvest_data_dir,
)

import xarray
from pyproj import Transformer
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import presto

regression = False
multilabel = False
num_outputs = 1
start_month = 1
num_timesteps = 1

path_to_flood_images = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/S2GeodnHand6Bands"
path_to_labels = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/data/data/flood_events/HandLabeled/LabelHand"
train_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_train_data_S2_geodn.txt"
valid_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_valid_data_S2_geodn.txt"
test_split = "/dccstor/geofm-finetuning/flood_mapping/sen1floods11/splits/splits/flood_handlabeled/flood_test_data_S2_geodn.txt"


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


def truncate_timesteps(self, x):
    if (self.num_timesteps is None) or (x is None):
        return x
    else:
        return x[:, : self.num_timesteps]


def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
    # TODO - where are these controlled?
    lr, num_grad_steps, k = 0.001, 250, 10
    model = self._construct_finetuning_model(pretrained_model)

    # TODO - should this be more intelligent? e.g. first learn the
    # (randomly initialized) head before modifying parameters for
    # the whole model?
    opt = SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="mean")
    batch_mask = self._mask_to_batch_tensor(mask, k)

    for i in range(num_grad_steps):
        if i != 0:
            model.train()
            opt.zero_grad()

        train_x, train_dw, latlons, train_y = self.dataset.sample(k, deterministic=False)
        preds = model(
            self.truncate_timesteps(
                torch.from_numpy(S1_S2_ERA5_SRTM.normalize(train_x)).to(device).float()
            ),
            mask=self.truncate_timesteps(batch_mask),
            dynamic_world=None,
            latlons=None,
            month=None
        ).squeeze(dim=1)
        loss = loss_fn(preds, torch.from_numpy(train_y).to(device).float())

        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(
        self,
        finetuned_model: Union[FineTuningModel],
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
) -> Dict:
    if isinstance(finetuned_model):
        assert isinstance(pretrained_model, (Mosaiks1d, Seq2Seq))

    with tempfile.TemporaryDirectory() as results_dir:
        for test_id, test_instance, test_dw_instance in self.dataset.test_data(max_size=10000):
            savepath = Path(results_dir) / f"{test_id}.nc"

            test_x = self.truncate_timesteps(
                torch.from_numpy(S1_S2_ERA5_SRTM.normalize(test_instance.x)).to(device).float()
            )
            # mypy fails with these lines uncommented, but this is how we will
            # pass the other values to the model
            test_latlons_np = np.stack([test_instance.lats, test_instance.lons], axis=-1)
            test_latlon = torch.from_numpy(test_latlons_np).to(device).float()
            test_dw = self.truncate_timesteps(
                torch.from_numpy(test_dw_instance.x).to(device).long()
            )
            batch_mask = self.truncate_timesteps(
                self._mask_to_batch_tensor(mask, test_x.shape[0])
            )

            if isinstance(finetuned_model, FineTuningModel):
                finetuned_model.eval()
                preds = (
                    finetuned_model(
                        test_x,
                        dynamic_world=test_dw,
                        mask=batch_mask,
                        latlons=test_latlon,
                        month=self.start_month,
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
                        month=self.start_month,
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
    return {f"{self.name}: {prefix}_{key}": val for key, val in combined_results.items()}


def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
) -> Dict:
    for x in model_modes:
        assert x in ["finetune", "Regression", "Random Forest"]
    results_dict = {}
    if "finetune" in model_modes:
        model = self.finetune(pretrained_model, mask)
        results_dict.update(self.evaluate(model, None, mask))

    return results_dict


# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

treesat_folder = presto.utils.data_dir / "treesat"
assert treesat_folder.exists()

# this folder should exist once the s2 file from zenodo has been unzipped
s2_data_60m = treesat_folder / "s2/60m"
assert s2_data_60m.exists()

TREESATAI_S2_BANDS = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]

SPECIES = ["Abies_alba", "Acer_pseudoplatanus"]

# takes a (6, 6) treesat tif file, and returns a
# (9,1,18) cropharvest eo-style file (with all bands "masked"
# except for S1 and S2)
INDICES_IN_TIF_FILE = list(range(0, 6, 2))

with (treesat_folder / "train_filenames.lst").open("r") as f:
    train_files = [line for line in f if (line.startswith(SPECIES[0]) or line.startswith(SPECIES[1]))]
with (treesat_folder / "test_filenames.lst").open("r") as f:
    test_files = [line for line in f if (line.startswith(SPECIES[0]) or line.startswith(SPECIES[1]))]

print(f"{len(train_files)} train files and {len(test_files)} test files")


def process_images(filenames):
    arrays, masks, latlons, image_names, labels, dynamic_worlds = [], [], [], [], [], []

    for filename in tqdm(filenames):
        tif_file = xarray.open_rasterio(s2_data_60m / filename.strip())
        crs = tif_file.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        for x_idx in INDICES_IN_TIF_FILE:
            for y_idx in INDICES_IN_TIF_FILE:
                # firstly, get the latitudes and longitudes
                x, y = tif_file.x[x_idx], tif_file.y[y_idx]
                lon, lat = transformer.transform(x, y)
                latlons.append(torch.tensor([lat, lon]))

                # then, get the eo_data, mask and dynamic world
                s2_data_for_pixel = torch.from_numpy(tif_file.values[:, x_idx, y_idx].astype(int)).float()
                s2_data_with_time_dimension = s2_data_for_pixel.unsqueeze(0)
                x, mask, dynamic_world = presto.construct_single_presto_input(
                    s2=s2_data_with_time_dimension, s2_bands=TREESATAI_S2_BANDS
                )
                arrays.append(x)
                masks.append(mask)
                dynamic_worlds.append(dynamic_world)

                labels.append(0 if filename.startswith("Abies") else 1)
                image_names.append(filename)

    return (torch.stack(arrays, axis=0),
            torch.stack(masks, axis=0),
            torch.stack(dynamic_worlds, axis=0),
            torch.stack(latlons, axis=0),
            torch.tensor(labels),
            image_names,
            )


train_data = process_images(train_files)
test_data = process_images(test_files)

batch_size = 64

pretrained_model = presto.Presto.load_pretrained()
pretrained_model.eval()

# the treesat AI data was collected during the summer,
# so we estimate the month to be 6 (July)
month = torch.tensor([6] * train_data[0].shape[0]).long()

dl = DataLoader(
    TensorDataset(
        train_data[0].float(),  # x
        train_data[1].bool(),  # mask
        train_data[2].long(),  # dynamic world
        train_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)

features_list = []
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        encodings = (
            pretrained_model.encoder(
                x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
            )
                .cpu()
                .numpy()
        )
        features_list.append(encodings)
features_np = np.concatenate(features_list)

# finetuning
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        finetuning_model = pretrained_model.construct_finetuning_model(num_outputs=1)
        x, mask, dynamic_world = presto.construct_single_presto_input(
            s2_bands=["B2", "B3", "B4"]
            # s2_bands = ["B2", "B3", "B4", "B8A", "B11", "B12"]
        )
        predictions = finetuning_model(x, mask)
print(predictions)

# to make a randomly initialized encoder-decoder model
encoder_decoder = Presto.construct()
# alternatively, the pre-trained model can also be loaded
encoder_decoder = Presto.load_pretrained()

# to add a linear transformation to the encoder's output for finetuning
finetuning_model = encoder_decoder.construct_finetuning_model(num_outputs=1)

x, mask, dynamic_world = presto.construct_single_presto_input(
    s2_bands=["B2", "B3", "B4"]
    # s2_bands = ["B2", "B3", "B4", "B8A", "B11", "B12"]
)

predictions = finetuning_model(x, mask)

print(predictions)
