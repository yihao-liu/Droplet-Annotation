
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nd2
import cv2 as cv
from skimage.morphology import disk
from skimage.filters import rank
from preprocessing.raw_image_reader import get_image_as_ndarray
from pathlib import Path
from tqdm.auto import tqdm
from data_creation.droplet_retriever import create_dataset_cell_enhanced_from_ndarray, resize_patch
from data_creation.cell_detector import cell_detector
from data_creation.manual_circle_hough import manual_circle_hough
import argparse

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")
RESULT_PATH = Path(DATA_PATH / "05_results")
EXPERIMENT_PATH = Path(PROJECT_PATH / "experiments")

parser = argparse.ArgumentParser(
    description="Generate droplets with all channels.")
parser.add_argument("image_name", type=str,
                    help="Name of the image for processing")
parser.add_argument("-minr", "--radius_min", type=int, default=12,
                    help="Minimum radius of droplets to be detected in pixels. Default is 12")
parser.add_argument("-maxr", "--radius_max", type=int, default=25,
                    help="Maximum radius of droplets to be detected in pixels. Default is 25")
args = parser.parse_args()


image_name = args.image_name


def generate_output_from_ndarray(input_image, output_string_droplets, output_string_cells, refine, optional_output_directory, optional_output, radius_min=12, radius_max=25):
    nr_frames = input_image.shape[0]
    nr_channels = input_image.shape[1]
    channel_index = {"BF": 4, "DAPI": 0}
    BF = channel_index["BF"]  # output 4
    DAPI = channel_index["DAPI"]  # output 0
    droplets = []
    cells_dict = []
    for frame_nr in tqdm(range(nr_frames)):
        # for frame_nr in [8]:
        dapi_channel = input_image[frame_nr, DAPI, :, :]
        bf_channel = input_image[frame_nr, BF, :, :]
        visualization_channel = np.zeros(bf_channel.shape, dtype=np.float32)

        # cv.imshow("test", bf_channel[0: 0 + 1000, 0: 0 + 1000])
        # cv.waitKey(0)
        # dapi_channel = dapi_channel[0: 0 + 1000, 0: 0 + 1000]
        # bf_channel = bf_channel[0: 0 + 1000, 0: 0 + 1000]

        # print(bf_channel.shape)
        # print(dapi_channel.shape)

        circles_in_frame = manual_circle_hough(
            bf_channel, refine, bf_is_inverted=True, radius_min=radius_min, radius_max=radius_max)

        # cells_mask, cells_intensities, cells_persistencies, squashed_cells_intensities, squashed_cells_persistencies = cell_detector(dapi_channel, bf_channel, circles_in_frame)
        cells_mask, cells_intensities, cells_persistencies = cell_detector(
            dapi_channel, bf_channel, circles_in_frame)

        intensities_vector = cells_intensities[cells_mask == 1.0]
        persistence_vector = cells_persistencies[cells_mask == 1.0]

        intens_thresh = np.quantile(intensities_vector, 0.2)
        presis_thresh = np.quantile(persistence_vector, 0.2)

        visualization_channel = cv.morphologyEx(
            cells_mask, cv.MORPH_DILATE, np.ones((3, 3)))

        # assert(False)

        # cv.imshow("test", visualization_channel)
        # cv.waitKey(0)

        cell_id_counter = 0
        for id, circ in tqdm(enumerate(circles_in_frame)):
            center = np.asarray([circ[0], circ[1]])
            radius = circ[2]
            patch_x = (max(int(center[0]) - radius - 2, 0),
                       min(int(center[0]) + radius + 2, cells_mask.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 2, 0),
                       min(int(center[1]) + radius + 2, cells_mask.shape[1] - 1))
            local_cells_mask = cells_mask[patch_x[0]
                : patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_intens = cells_intensities[patch_x[0]
                : patch_x[1], patch_y[0]: patch_y[1]]
            local_cells_pers = cells_persistencies[patch_x[0]
                : patch_x[1], patch_y[0]: patch_y[1]]
            # local_squashed_cells_intens = squashed_cells_intensities[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            # local_squashed_cells_pers = squashed_cells_persistencies[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_mask = np.zeros(local_cells_mask.shape)
            center_in_patch = center - \
                np.asarray([max(int(center[0]) - radius - 2, 0),
                           max(int(center[1]) - radius - 2, 0)])
            cv.circle(local_mask, np.flip(center_in_patch), radius, 1.0, -1)
            local_cells_mask = local_cells_mask * local_mask
            local_cells_intens = local_cells_intens * local_mask
            local_cells_pers = local_cells_pers * local_mask
            # local_squashed_cells_intens = local_squashed_cells_intens * local_mask
            # local_squashed_cells_pers = local_squashed_cells_pers * local_mask
            # local_bf = bf_channel[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            # local_bf = local_bf / local_bf.max()
            # cv.circle(local_bf, np.flip(center_in_patch), radius, 1.0 , 1)
            # cv.imshow("test", local_bf)
            # cv.waitKey(0)

            nr_cells_estimated = np.sum(np.logical_and(
                (local_cells_pers > presis_thresh), (local_cells_intens > intens_thresh)))
            cv.circle(visualization_channel, np.flip(center), radius, 1.0, 1)
            droplets.append({"droplet_id": id, "frame": frame_nr,
                            "center_row": circ[0], "center_col": circ[1], "radius": circ[2], "nr_cells": nr_cells_estimated})
            cell_coords = np.transpose(np.asarray(
                np.where(local_cells_mask != 0.0)))
            for coord in cell_coords:
                global_center = coord + \
                    np.asarray([max(int(center[0]) - radius - 2, 0),
                               max(int(center[1]) - radius - 2, 0)])
                cells_dict.append({"cell_id": cell_id_counter,
                                   "droplet_id": id,
                                   "frame": frame_nr,
                                   "center_row": global_center[0],
                                   "center_col": global_center[1],
                                   "intensity_score": local_cells_intens[coord[0], coord[1]],
                                   "persistence_score": local_cells_pers[coord[0], coord[1]]})
                cell_id_counter = cell_id_counter + 1
        if optional_output:
            to_display = np.float32(np.transpose(np.asarray([visualization_channel * 1, (bf_channel - bf_channel.min()) / (bf_channel.max(
            ) - bf_channel.min()), 1.0 * (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())]), [1, 2, 0]))
            cv.imwrite(optional_output_directory +
                       'detection_visualization_frame_' + str(frame_nr) + '.tiff', to_display)
    droplet_df = pd.DataFrame(droplets)
    droplet_df.to_csv(output_string_droplets, index=False)

    cell_df = pd.DataFrame(cells_dict)
    cell_df.to_csv(output_string_cells, index=False)


def save_droplet_images(dataset: np.ndarray, image_name: str, DROPLET_PATH: Path) -> None:
    folder_path = Path(DROPLET_PATH / image_name)

    try:
        os.mkdir(folder_path)
    except FileExistsError as _:
        pass

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            # Changed patch size from 100x100 to 64x64 as 100x100 is just way too big considering the droplets are ca 40 pixels across
            patch = np.float64(resize_patch(dataset[i][j]['patch'], 64))
            patch[patch == 0.0] = np.nan
            thresh = np.nanquantile(patch, 0.5, axis=(1, 2))
            patch[np.isnan(patch)] = 0.0
            patch = np.uint16(np.clip((patch - thresh[:, None, None]) / (
                2**16 - 1 - thresh[:, None, None]), 0, 1.0) * (2**16 - 1))
            patch[np.isnan(patch)] = 0.0
            np.save(
                Path(folder_path / ("f" + str(i) + "_d" + str(j).zfill(4))), patch)


def preprocess_alt_franc_all_frames(image_path: str) -> np.ndarray:
    image = get_image_as_ndarray(
        [0], ['BF', 'DAPI'], image_path, allFrames=True, allChannels=True)
    channel_index = {"BF": 4, "DAPI": 0}
    BF = channel_index["BF"]  # output 4
    DAPI = channel_index["DAPI"]  # output 0
    # For each frame, preprocess channels inplace
    # but why do we need to do this?
    image[:, BF, :, :] = np.uint16(2**16 - (np.int32(image[:, BF, :, :]) + 1))

    kernel = np.ones((3, 3))

    for frame in image:
        # print(np.median(frame[4]))
        bf_chan = np.float64(frame[BF, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) /
                          (bf_chan_high - bf_chan_low), 0.0, 1.0)
        # I don't know why the second normalizatio is needed
        pullback_min = bf_chan.min()
        pullback_max = bf_chan.max()
        bf_pullback = (bf_chan - pullback_min) / (pullback_max - pullback_min)
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)
        # I do not get why choose quantile 0.5 here, since then values lower than 0.5 will be clipped to 0
        bf_pullback = np.clip((bf_pullback - np.quantile(bf_pullback, 0.5)) /
                              (1.0 - np.quantile(bf_pullback, 0.5)), 0.0, 1.0)
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)

        equalized = rank.equalize(bf_pullback, footprint=disk(10)) / 255.0
        # cv.imshow('test', equalized[:1000, :1000])
        # cv.waitKey(0)
        # why do we need to do this?
        bf_pullback = bf_pullback * equalized
        # cv.imshow('test', bf_pullback[:1000, :1000])
        # cv.waitKey(0)
        smoothed = cv.GaussianBlur(bf_pullback, (3, 3), 0)
        frame[BF, :, :] = np.uint16(smoothed * (2**16 - 1))
    return image


# the original order of channels is DAPI, FITC, TRITC, Cy5, BF
# the required order of channels is BF, DAPI and the rest
# TODO: include the index of DAPI and BF
# for example: {"BF":4, "DAPI":0}
# channel_index = {"BF":4, "DAPI":0}
# channel_index["BF"] # output 4

def preprocess_alt_featextr_all_frames(image_path: str) -> np.ndarray:
    image = get_image_as_ndarray(
        [0], ['BF', 'DAPI'], image_path, allFrames=True, allChannels=True)
    channel_index = {"BF": 4, "DAPI": 0}
    BF = channel_index["BF"]  # output 4
    DAPI = channel_index["DAPI"]  # output 0
    image[:, BF, :, :] = np.uint16(2**16 - (np.int32(image[:, BF, :, :]) + 1))

    # For each frame, preprocess channels inplace
    for frame in tqdm(image):
        # Brightfield preprocessing
        bf_chan = np.float64(frame[BF, :, :])
        bf_chan_low = np.quantile(bf_chan, 0.1)
        bf_chan_high = np.quantile(bf_chan, 0.995)
        bf_chan = np.clip((bf_chan - bf_chan_low) /
                          (bf_chan_high - bf_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(
            np.uint8(bf_chan * 255), 2 * 50 + 1) / 255.0)
        img_mediansharpened = np.clip(bf_chan - img_medianblurred, 0.0, 1.0)
        equalized_bf = rank.equalize(
            img_mediansharpened, footprint=disk(10)) / 255.0
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.5)
        img_mediansharpened[img_mediansharpened >
                            thresh] = bf_chan[img_mediansharpened > thresh]

        frame[BF] = cv.GaussianBlur(
            np.uint16(img_mediansharpened * (2**16 - 1)), (3, 3), 0)
        # cv.imshow('test', bf_chan[:1000, :1000])
        # cv.waitKey(0)
        # cv.imshow('test', frame[0, :1000, :1000])
        # cv.waitKey(0)

        # DAPI preprocessing
        dapi_chan = np.float64(frame[DAPI, :, :])
        dapi_chan_low = np.quantile(dapi_chan, 0.8)
        dapi_chan = np.clip((dapi_chan - dapi_chan_low) /
                            ((2**16 - 1) - dapi_chan_low), 0.0, 1.0)
        img_medianblurred = np.float64(cv.medianBlur(
            np.uint8(dapi_chan * 255), 2 * 20 + 1) / 255.0)
        img_mediansharpened = np.clip(dapi_chan - img_medianblurred, 0.0, 1.0)
        equalized_dapi = rank.equalize(
            img_mediansharpened, footprint=disk(10)) / 255.0
        # I am not sure they are wrong here, but I think they should use equalized_dapi instead of equalized_bf
        img_mediansharpened = img_mediansharpened * equalized_bf
        thresh = np.quantile(img_mediansharpened, 0.8)
        img_mediansharpened[img_mediansharpened >
                            thresh] = dapi_chan[img_mediansharpened > thresh]

        frame[DAPI] = np.uint16(img_mediansharpened * (2**16 - 1))
        # cv.imshow('test', dapi_chan[:1000, :1000])
        # cv.waitKey(0)
        # cv.imshow('test', frame[1, :1000, :1000])
        # cv.waitKey(0)
    image[np.isnan(image)] = 0.0
    return image


image_path = os.path.join(RAW_PATH, image_name + ".nd2")
image = get_image_as_ndarray(
    [0], ['BF', 'DAPI'], image_path, allFrames=True, allChannels=True)

# Read the image with preprocessing (not for the feature extraction)
processed_image = preprocess_alt_franc_all_frames(image_path)

# Read the image with preprocessing (for the feature extraction)
processed_image_feature = preprocess_alt_featextr_all_frames(image_path)

# Generate preprocessed droplet images
radius_min = args.radius_min
radius_max = args.radius_max
print("Detecting droplets and cells...")
droplet_feature_path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
cell_feature_path = Path(FEATURE_PATH / f"cells_{image_name}.csv")
# generate_output_from_ndarray(preprocessed_image, droplet_feature_path, cell_feature_path, True, str(PREPROCESSED_PATH) + "/", True)
generate_output_from_ndarray(processed_image, droplet_feature_path, cell_feature_path,
                             True, "", False, radius_min=radius_min, radius_max=radius_max)

# Now we have two situations:
# 1. if we only have the coordinates and the diameter of the droplets, we should use the processed image for visualization (what we have from old script)
# 2. if we have the droplet id, we can use the droplet images for visualization (what we have from the new python script)
print("Creating droplet images...")
droplet_images_dataset = create_dataset_cell_enhanced_from_ndarray(
    [0], processed_image_feature, droplet_feature_path, cell_feature_path, allFrames=True, buffer=-2, suppress_rest=True, suppression_slack=-3)
save_droplet_images(droplet_images_dataset, image_name, DROPLET_PATH)

# TODO: do we really need this fulldataset?
fullset_path = Path(FEATURE_PATH / f"fulldataset_{image_name}.npy")
np.save(fullset_path, droplet_images_dataset)

print("Generating droplet images finished.")
