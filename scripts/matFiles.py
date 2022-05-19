from pathlib import Path
from typing import List, Union, Counter

import numpy as np
from scipy.io import loadmat, savemat


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


class Merge_MatFiles(object):
    def __init__(self, mat_files: List[Union[str]], new_matFile: str):
        self.mat_files = mat_files
        self.matDict = dict()

        matDictList = self.get_matDicts(self.mat_files)
        self.matDict = self.intersect_matDicts(matDictList)
        save_mat_file(self.matDict)

    @staticmethod
    def get_matDicts(files):
        matDictList = list()
        for file in files:
            matDictList.append(read_mat_file(file))
        return matDictList

    @staticmethod
    def intersect_matDicts(matDictsList: list):
        matDict = dict()
        for mat in matDictsList:
            for key, value in mat.items():
                if key not in matDict.keys():
                    matDict[key] = value
                else:
                    if key not in ['imnames', 'txt', 'wordBB', 'charBB']:
                        continue
                    else:
                        if not isinstance(value, np.ndarray):
                            raise Exception(f'Mat File Key {key} value is Not Numpy')
                        #todo:
                        # if not value.shape == (1, ...):
                        #     raise Exception(f'Key {key} value not of shape (1, ...) but {value.shape}')
                        matDict[key] = np.concatenate((matDict[key], value), axis=1)
        return matDict


if __name__ == '__main__':
    root = '/home/mansoor/Projects/Craft-Training/'
    matFiles = [root + 'dataset/craft/gt.mat', root + 'dataset/craft-english/gt.mat']
    new_matFile = root + 'dataset/EnglishUrdu_synthDocs/gt.mat'
    Merge_MatFiles(matFiles, )
    # file = read_mat_file( matFiles[1])
    # pass