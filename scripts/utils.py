import glob
import os
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any
import subprocess as sp
from natsort import natsorted
from scipy.io import loadmat, savemat
import cv2
import random

import yaml
import json
from omegaconf import OmegaConf
import pickle

from pdf2image import convert_from_path


def read_mat_file(filePath: str = '') -> dict:
    if not filePath:
        raise Exception(f"Empty File Path for Mat File !")
    filePath = Path(filePath)
    if not filePath.exists():
        raise Exception(f"Mat File {filePath} Doesn't Exists !")
    if filePath.suffix != '.mat':
        raise Exception(f"Incorrect Mat File {filePath} !")
    mat = loadmat(filePath.as_posix())
    return mat


def save_mat_file(filePath: str, matDict: dict):
    if not filePath:
        raise Exception(f"Empty File Path for Mat File !")
    if not isinstance(matDict, dict):
        raise Exception("Mat File Not a Dictionary !")
    filePath = Path(filePath)
    if filePath.suffix != '.mat':
        raise Exception(f"Incorrect Mat File {filePath} !")
    if filePath.exists():
        raise Exception(f"Mat File Already Exists {filePath} !")
    filePath.parent.mkdir(parents=True, exist_ok=True)
    savemat(filePath.as_posix(), matDict)
    print(f"Mat File Saved at {filePath}")


def collect_file_paths(path: str, suffixes: Union[List[str], List[str]] = None,
                       flatten: bool = True, sort: bool = True) -> list:
    """Returns list of file_paths with matching suffixes"""
    suffixes = [] if suffixes is None else suffixes
    collected_file_paths = list()
    if not path:
        return collected_file_paths

    try:
        search_paths = os.listdir(path)
    except NotADirectoryError:
        search_paths = [path]
    except FileNotFoundError:
        search_paths = []

    for search_path in search_paths:
        abs_search_path = os.path.join(path, search_path)
        # If search path is a directory, then,
        # collect file paths from it and append to collection
        if os.path.isdir(abs_search_path):
            file_paths = collect_file_paths(
                path=abs_search_path,
                suffixes=suffixes,
                flatten=flatten
            )
            collected_file_paths += file_paths if flatten or not file_paths else [file_paths]
        # If search path is a file, then,
        # append directly to collection
        else:
            file_name = Path(abs_search_path).name
            if suffixes and not file_name.endswith(tuple(suffixes)):
                collected_file_paths += []
            else:
                collected_file_paths += [abs_search_path]
    if sort:
        collected_file_paths = natsorted(collected_file_paths)
    return collected_file_paths


def meta_img_rotation(imgDir: str):
    im_file_paths = collect_file_paths(imgDir, ["jpg", "jpeg", "png"])

    for im_file_path in im_file_paths:
        sp.run(["exiftool", "-all=", f"{im_file_path}", "-overwrite_original"])
        # sp.run(["convert", "-rotate", "+180", f"{im_file_path}", f"{im_file_path}"])
    # convert -rotate "-90" in.jpg out.jpg


def pdf2images(pdfFiles: List[Union[str]], imgDir: str, outputImgFmt: str = 'png', outputImgSize: Tuple[int] = (1500, 1000)):
    """
    Note:
        1. pdfDir should ONLY have pdf files with any name or extension.
    """
    imgDir = Path(imgDir)

    if not imgDir.exists():
        imgDir.mkdir(parents=True, exist_ok=True)

    for pdfFile in pdfFiles:
        pdfFile = Path(pdfFile)
        imgTemplateDir = imgDir / pdfFile.stem
        if imgTemplateDir.exists():
            # raise Exception(f"Image Template Dir Already Exists : {imgTemplateDir.as_posix()}")
            pass
        else:
            imgTemplateDir.mkdir(parents=True, exist_ok=True)

        pgImgs = convert_from_path(pdf_path=pdfFile.as_posix(),
                                   # output_folder=imgTemplateDir.as_posix(), # lets save using custome image names.
                                   fmt=outputImgFmt,
                                   size=outputImgSize if outputImgSize else None,
                                   jpegopt={'quality': 100, 'progressive': True, 'optimize': True})
        print(f"Writing Images in {imgTemplateDir} ...  ")
        for imgNo, img in enumerate(pgImgs):
            file = imgTemplateDir.as_posix() + '/' + str(imgNo) + '.' + outputImgFmt
            img.save(file)
            print(f"Writing Image {file}")


def clip_or_remove_bboxes(bboxes: List[Union[int, int, int, int, str]], rows: int, cols: int,
                          remove_bboxes: bool = True) -> List[Union[int, int, int, int, str]]:
    """
    Params:
        bboxes -->   [ [x_min, y_min, x_max, y_max, 'Text'],
                        [x_min, y_min, x_max, y_max, 'Text'],  ... ]
    Objective:
        Either remove the bboxes with out-of-bound coordinates (Recommended) OR clip them to max/min of image shape

    Issues:
        np.asarray --> this operation ultimately converts all the datatypes into single datatypes.
                        because out {bboxes} have both str and int values therefore asarray function
                        converts int to str values. Even converting types of few columns, using astypes,
                        would not be helpfull.
        np.concatenate --> this function will behave same as above. If we try to cocatenate or stack int values
                        matrix and str matrix then the resulting matrix dataype will be str only.
                        Thus, to avoid this issue we convert the matrix into list and merge them.
    """

    bboxes = np.asarray(bboxes)
    # separate bboxes labels and coordinates:
    labels = bboxes[:, -1]
    coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
    if not remove_bboxes:
        """
        Note: this approach is not best and reliable since you may get the error:
            >> y_max is less than or equal to y_min for bbox
            Therefore, It highly recommended to remove the out-of-bound bboxes.
        """
        coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / cols
        coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / rows
        coordinates = np.clip(coordinates, 0, 1)
        coordinates[:, [0, 2]] = coordinates[:, [0, 2]] * cols
        coordinates[:, [1, 3]] = coordinates[:, [1, 3]] * rows
    else:
        # find out the rows/bboxes that have coordinates out of the range [0, 1]
        outOfBond_rows = list(set(np.where((coordinates[:, [0, 2]] > cols) | (coordinates[:, [0, 2]] < 0) |
                                           (coordinates[:, [1, 3]] > rows) | (coordinates[:, [1, 3]] < 0))[0]))
        coordinates = np.delete(coordinates, outOfBond_rows, 0)
        labels = np.delete(labels, outOfBond_rows, 0)
    coordinates, labels = coordinates.tolist(), labels.tolist()
    [coord.append(label) for coord, label in zip(coordinates, labels)]
    return coordinates


def resize_bboxes(current_bboxes: List[List[Union[float, str]]], currentImgSize: Tuple[int, int],
                  originalImgSize: Tuple[int, int]) -> List[List[Union[float, str]]]:
    bboxes = np.asarray(current_bboxes)
    labels = bboxes[:, -1]   # separate bboxes labels and coordinates:
    coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / currentImgSize[1]
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / currentImgSize[0]
    coordinates = np.clip(coordinates, 0, 1)
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]] * originalImgSize[1]
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]] * originalImgSize[0]
    coordinates, labels = coordinates.tolist(), labels.tolist()
    [coord.append(label) for coord, label in zip(coordinates, labels)]
    return coordinates


def normalize_bboxes(current_bboxes: List[List[Union[float, str]]], currentImgSize: Tuple[int, int]) \
        -> List[List[Union[float, str]]]:
    bboxes = np.asarray(current_bboxes)
    labels = bboxes[:, -1]  # separate bboxes labels and coordinates:
    coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / currentImgSize[1]
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / currentImgSize[0]
    coordinates = np.clip(coordinates, 0, 1)
    coordinates, labels = coordinates.tolist(), labels.tolist()
    [coord.append(label) for coord, label in zip(coordinates, labels)]
    return coordinates


def resize_images(imgDir: str, resizeTo: Tuple[int, int], imgFmts: List[Union[str]] = [], overwrite: bool = True):
    #todo: test this function ....
    imgDir = Path(imgDir)
    if not imgDir.exists():
        raise Exception("Image Dir Doesn't Exists !")
    for file in imgDir.iterdir():
        if imgFmts and file.suffix not in imgFmts:
            continue
        if overwrite:
            img = cv2.imread(file.as_posix())
            img = cv2.resize(img, resizeTo)
            cv2.imwrite(file.as_posix(), img)
            print(f"Overwriting Image at {file.as_posix()}")
        else:
            writeToDir = imgDir.parent / 'Resized_Images'
            writeToDir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(file.as_posix())
            img = cv2.resize(img, resizeTo)
            writefile = writeToDir / file.name
            cv2.imwrite(writefile.as_posix(), img)
            print(f"Writing Image {writefile}")

def plot_bbox(img: np.ndarray, bboxes: List[List[Union[int, int, int, int, str]]],
              bbox_color: Tuple[int, int, int] = (255, 0, 0),
              bbox_thickness: int = 2) -> np.ndarray:
    for bbox in bboxes:
        # bbox --> [x_min, y_min, x_max, y_max, 'Text']
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        # cv2.rectange:  img -> HWC & start_points/end_points --> (int, int)
        img = cv2.rectangle(img, start_point, end_point, bbox_color, bbox_thickness)
    return img


def find_min_max_coordinates(coordinates: List) -> Dict:
    coordinates = np.asarray(coordinates).astype(int)
    x_indices = np.arange(0, len(coordinates), 2)
    y_indices = np.arange(1, len(coordinates), 2)
    x_coordinates, y_coordinates = coordinates[x_indices], coordinates[y_indices]
    x_min, y_min, x_max, y_max = min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)
    return dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def search_file(searchDir: str, pattern: str) -> str:
    file = searchDir + '/' + pattern
    files = glob.glob(file)
    if len(files) == 0:
        raise Exception('No File of Such Pattern Found.')
    elif len(files) > 1:
        raise Exception('Multiple Files of Similar Pattern Found.')
    return files[0]


def split_train_test_indices(totLength: int, perVal: float)-> Dict[str, Union[set, list]]:
    lenVal = int(perVal * totLength)
    valIndices = random.sample(range(totLength), lenVal)
    trainIndices = set(range(totLength)) - set(valIndices)
    return dict(trainIndices=trainIndices, valIndices=valIndices)


def rename_files(dirPath: str):
    dirPath = Path(dirPath)
    for file in dirPath.iterdir():
        source = file
        destination = dirPath / file.name.replace('gt_', '').replace('img_', '')
        os.rename(source, destination)


def pascalVoc_to_Coco128(bboxes: List[List[Union[float, str]]], imgWidth: int, imgHeight=int) -> \
        List[List[Union[float, str]]]:
    """
    Input (pascal) Format:   [ [x_min, y_min, x_max, y_max, 'Text'], [], ....]
    Output (coco128) Format: [ [0, x_center, y_center, width, height], [], ....]
    """
    bboxes = np.asarray(bboxes)
    labels = bboxes[:, -1]  # separate bboxes labels and coordinates:
    labels = np.asarray([0 for _ in labels])

    coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / int(imgWidth)
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / int(imgHeight)

    bboxes_width = coordinates[:, 2] - coordinates[:, 0]
    bboxes_height = coordinates[:, 3] - coordinates[:, 1]
    x_center = coordinates[:, 0] + (bboxes_width / 2)
    y_center = coordinates[:, 1] + (bboxes_height / 2)

    x_center, y_center, bboxes_width, bboxes_height, labels = \
        x_center[..., None], y_center[..., None], bboxes_width[..., None], bboxes_height[..., None], labels[..., None]

    cocoBboxes = np.concatenate((labels, x_center, y_center, bboxes_width, bboxes_height), axis=1)

    return cocoBboxes


def read_txt(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def writeNdarryToText(file_path: str, array: np.ndarray):
    """Write data to text file"""
    file_path = apply_extension(file_path, "txt")
    ensure_dir_exists(file_path)
    fileHandler = open(file_path, 'w')
    for bbox in array:
        line = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n"
        fileHandler.write(line)
    print(f"Wrote File {file_path}")


def get_image_dim(img: np.ndarray) -> Dict:
    img_dims = len(img.shape)
    if img_dims == 2:
        height, width = img.shape
        channels = 1
    elif img_dims == 3:
        height, width, channels = img.shape
    else:
        height = width = channels = None
    return dict(height=height, width=width, channels=channels)

# ----------------   M.I.Baig  ------------------


def read_pickle(file_path: str) -> Any:
    """Return data from pickle file"""
    with open(file_path, "rb") as f:
        try:
            file_data = pickle.load(f)
            return file_data
        except pickle.UnpicklingError as e:
            print(e)


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    tag='tag:yaml.org,2002:python/tuple',
    constructor=PrettySafeLoader.construct_python_tuple)


def write_txt(file_path: str, data: str):
    """Write data to text file"""
    file_path = apply_extension(file_path, "txt")
    ensure_dir_exists(file_path)
    with open(file_path, "w") as f:
        f.write(data)


def read_yaml(file_path: str, as_dict_conf: bool = False):
    """Return data from yaml file"""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            file_data = yaml.load(f, PrettySafeLoader)
            file_data = file_data if file_data is not None else dict()
            file_data = OmegaConf.create(file_data) if as_dict_conf else file_data
            return file_data
        except yaml.YAMLError as e:
            print(e)


def write_yaml(file_path: str, data: Any, sort_keys: bool = True) -> None:
    """Write data to yaml file"""
    file_path = apply_extension(file_path, "yaml")
    ensure_dir_exists(file_path)
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def read_json(file_path: str) -> Any:
    """Return data from json file"""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            file_data = json.load(f)
            return file_data
        except json.JSONDecodeError as e:
            print(e)


def write_json(file_path: str, data: Any, sort_keys: bool = True) -> None:
    """Write data to json file"""
    file_path = apply_extension(file_path, "json")
    ensure_dir_exists(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f, sort_keys=sort_keys)


def ensure_dir_exists(path: str) -> None:
    """Create parent directories to path"""
    ppath = Path(path)
    dir_ppath = ppath if not ppath.suffix else ppath.parent
    dir_ppath.mkdir(parents=True, exist_ok=True)


def apply_extension(file_path: str, extension: str) -> str:
    """Return file_path with extension"""
    file_ppath = Path(file_path)
    if file_ppath.suffix != extension:
        file_ppath = file_ppath.with_suffix(".{}".format(extension))
    return file_ppath.as_posix()


if __name__ == '__main__':
    """ Convert Pdf 2 Images:
    pdfs = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Dynamic-CharAnnots' \
           '/PDFs_Resized_Images_Dynamic_Annotations'
    imgs = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Dynamic-CharAnnots/Static&Dynamic_Char_Images'
    pdf2images(pdfs, imgs)
    """
    # MatFile = loadmat('/home/mansoor/Projects/Craft-Training/dataset/craft-english/gt.mat')
    # print()
    resize_images('/home/mansoor/Projects/Yolov5-nano-Word-Detector/runs/detect/Dataset_T',
                  resizeTo=(640, 640), overwrite=False)