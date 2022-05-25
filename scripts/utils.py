import glob
import os
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Dict
import subprocess as sp
from natsort import natsorted
from scipy.io import loadmat, savemat
import cv2
import random

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


def normalize_bboxes(current_bboxes: List[List[Union[float, str]]], currentImgSize: Tuple[int, int]) -> List[List[Union[float, str]]]:
    bboxes = np.asarray(current_bboxes)
    labels = bboxes[:, -1]  # separate bboxes labels and coordinates:
    coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
    coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / currentImgSize[1]
    coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / currentImgSize[0]
    coordinates = np.clip(coordinates, 0, 1)
    coordinates, labels = coordinates.tolist(), labels.tolist()
    [coord.append(label) for coord, label in zip(coordinates, labels)]
    return coordinates


def resize_images(imgDir: str, resizeTo: Tuple[int, int], imgFmts: List[Union[str]], overwrite: bool = True):
    #todo: test this function ....
    imgDir = Path(imgDir)
    if not imgDir.exists():
        raise Exception("Image Dir Doesn't Exists !")
    for file in imgDir.iterdir():
        if file.suffix in imgFmts:
            if overwrite:
                img = cv2.imread(file.as_posix())
                img = cv2.resize(img, resizeTo)
                cv2.imwrite(file.as_posix(), img)
                print(f"Overwriting Image at {file.as_posix()}")


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


if __name__ == '__main__':
    """ Convert Pdf 2 Images:
    pdfs = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Dynamic-CharAnnots' \
           '/PDFs_Resized_Images_Dynamic_Annotations'
    imgs = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Dynamic-CharAnnots/Static&Dynamic_Char_Images'
    pdf2images(pdfs, imgs)
    """
    # MatFile = loadmat('/home/mansoor/Projects/Craft-Training/dataset/craft-english/gt.mat')
    # print()
    split_train_test_indices(120, 0.14)