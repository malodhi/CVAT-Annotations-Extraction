from scripts.ParseAnnotations.CVAT.xmlPascalVoc import PascalVocXml
from scripts.utils import split_train_test_indices
from typing import List, Tuple, Union, Optional
import numpy as np
from pathlib import Path
import shutil


class ConvertDataSources(object):
    def __init__(self, datasetDirs: List[str], writeDataDir: str, val_per: float = 0.25):

        self.datasetDirs = datasetDirs
        self.writeDataDir = Path(writeDataDir)
        if not self.writeDataDir:
            self.writeDataDir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.val_per = val_per
        self.map_classes = dict(Text=1)

        self.imagesDir = self.writeDataDir / 'images'
        self.imagesDir.mkdir(parents=True, exist_ok=True)

        self.imagesTrainDir = self.imagesDir / 'train'
        self.imagesTrainDir.mkdir(parents=True, exist_ok=True)

        self.imagesValDir = self.imagesDir / 'val'
        self.imagesValDir.mkdir(parents=True, exist_ok=True)

        self.labelsDir = self.writeDataDir / 'labels'
        self.labelsDir.mkdir(parents=True, exist_ok=True)

        self.labelsTrainDir = self.labelsDir / 'train'
        self.labelsTrainDir.mkdir(parents=True, exist_ok=True)

        self.labelsValDir = self.labelsDir / 'val'
        self.labelsValDir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.datasetDirs)

    def __getitem__(self, index):
        self.datasetDir = self.datasetDirs[index]
        print(f"Converting Data From Directory ...  {self.datasetDir}")
        imgDir = self.datasetDir + '/images'
        gtFile = self.datasetDir + '/annotations.xml'
        allAnnots = PascalVocXml(self.datasetDir, imgDir, gtFile).annotations
        totFiles = len(allAnnots.keys())
        resultIndices = split_train_test_indices(totFiles, self.val_per)
        for item, (fileName, annots) in enumerate(allAnnots.items()):
            readimgFile = imgDir + '/' + fileName
            if item in resultIndices['trainIndices']:
                writeimgFile = self.imagesTrainDir.as_posix() + '/' + str(self.step) + '.' + fileName.split('.')[-1]
                writeTextFile = self.labelsTrainDir.as_posix() + '/' + str(self.step) + '.txt'
            elif item in resultIndices['valIndices']:
                writeimgFile = self.imagesValDir.as_posix() + '/' + str(self.step) + '.' + fileName.split('.')[-1]
                writeTextFile = self.labelsValDir.as_posix() + '/' + str(self.step) + '.txt'
            else:
                continue
            shutil.copy(readimgFile, writeimgFile)

            bboxes = self.convertBbox2Coco(**annots)
            self.writeBboxes(bboxes, writeTextFile)
            self.step += 1

    def convertBbox2Coco(self, bboxes: List[List[Union[float, str]]], width: str, height: str) -> np.ndarray:
        """
        bboxes: [ [x_min, y_min, x_max, y_max, label], ...]
        """
        bboxes = np.asarray(bboxes)
        labels = bboxes[:, -1]  # separate bboxes labels and coordinates:
        labels = np.asarray([self.map_classes[value] for value in labels])

        coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
        coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / int(width)
        coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / int(height)

        bboxes_width = coordinates[:, 2] - coordinates[:, 0]
        bboxes_height = coordinates[:, 3] - coordinates[:, 1]
        x_center = coordinates[:, 0] + (bboxes_width / 2)
        y_center = coordinates[:, 1] + (bboxes_height / 2)

        x_center, y_center, bboxes_width, bboxes_height, labels = \
            x_center[..., None], y_center[..., None], bboxes_width[..., None], bboxes_height[..., None], labels[..., None]

        cocoBboxes = np.concatenate((labels, x_center, y_center, bboxes_width, bboxes_height), axis=1)
        # cocoBboxes[:, [1, 3]] = cocoBboxes[:, [1, 3]].astype('float') / float(width)
        # cocoBboxes[:, [2, 4]] = cocoBboxes[:, [2, 4]].astype('float') / float(height)
        return cocoBboxes

    @staticmethod
    def writeBboxes(bboxes: np.ndarray, fileName: str):
        if bboxes.shape[1] != 5:
            raise Exception("BBoxes Shape Should be (:, 5) !")
        if Path(fileName).exists():
            print(f'Deleting Annotation Text File  ...  {fileName}')
            try:
                Path(fileName).unlink()
            except OSError as e:
                print("Error: %s : %s" % (fileName, e.strerror))
        f = open(fileName, 'w+')
        for bbox in bboxes:
            f.write(f"{int(0)} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
        f.close()
        print(f"Wrote File {fileName}")
        return


if __name__ == '__main__':
    dataset = ConvertDataSources(
        [
            '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Cvat-Words-Dataset/wla-1',
            '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Cvat-Words-Dataset/wla-3',
            '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Cvat-Words-Dataset/wla-8',
            '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Cvat-Words-Dataset/wla-9',
         ],
        '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Words'
    )
    list([_ for _ in dataset])