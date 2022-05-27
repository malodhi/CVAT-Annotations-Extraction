from pathlib import Path
import json
import matplotlib.pyplot as plt


from scripts.utils import *

# Source: https://rrc.cvc.uab.es/?ch=9&com=tasks


class ICDAR2017(object):
    def __init__(self, imgsDir: str, lblsDir: str):
        """
        dataDir:
            - test          -> images
            - test_gt       -> text annotation files
            - test_gt.json  -> json annotations file
            - train         -> images
            - train_gt      -> text annotation files
            - test_gt.json  -> json annotations file
        """
        self.imgsDir = Path(imgsDir)
        self.lblsDir = Path(lblsDir)
        self.imgFiles = [file for file in self.imgsDir.iterdir()]
        self.lblFiles = [file for file in self.lblsDir.iterdir()]
        if len(self.lblFiles) != len(self.imgFiles):
            raise Exception("Images & Labels length don't match !")

        self.annotations = self.readTextLbls(self.lblsDir)

    @staticmethod
    def readJsonLbls(filePath: str):
        data = read_json(filePath)
        annots = data.get('annots')
        renameAnnots = dict()
        # rename keys with respect to file names in image dir
        for fileName, annotations in annots.items():
            fileName = fileName.replace('gt_', '').replace('.txt', '')
            renameAnnots[fileName] = annotations
        return renameAnnots

    @staticmethod
    def readTextLbls(filesDir: Path):
        renameAnnots = dict()
        for fileName in filesDir.iterdir():
            data = read_txt(fileName.as_posix())
            fileName = fileName.stem.replace('gt_', '')
            bboxesText = list()
            for word in data:
                word = word.split(',')  # x1, y1, x2, y2, x3, y3, x4, y4, transcription
                min_max_coordinates = find_min_max_coordinates(word[0:8])  # todo: makde it fragile
                x_min, y_min, x_max, y_max = min_max_coordinates['x_min'], min_max_coordinates['y_min'], \
                                             min_max_coordinates['x_max'], min_max_coordinates['y_max']
                if x_min >= x_max or y_min >= y_max:
                    continue
                    # raise Exception("Bounding Boxes Not Synchronized !")
                bboxesText.append([x_min, y_min, x_max, y_max, 'text'])
            renameAnnots[fileName] = bboxesText
        return renameAnnots

    def writeToCoCo128(self):
        writeDir = self.lblsDir.parent.parent / ('CocoLabels/' + self.lblsDir.name)
        writeDir.mkdir(parents=True, exist_ok=True)
        for lblFile in self.lblsDir.iterdir():
            fileKey = lblFile.stem
            annots = self.annotations[fileKey]
            imgFile = search_file(self.imgsDir.as_posix(), str(fileKey) + '.*')
            img = plt.imread(imgFile)
            dimensions = get_image_dim(img)
            coco128Annots = pascalVoc_to_Coco128(annots, imgWidth=dimensions['width'], imgHeight=dimensions['height'])
            writeFile = writeDir / fileKey
            writeNdarryToText(writeFile, coco128Annots)

    def __len__(self):
        return len(self.lblFiles)

    def __getitem__(self, item):
        fileKey = self.lblFiles[item].stem
        annots = self.annotations[fileKey]
        imgFile = search_file(self.imgsDir.as_posix(), str(fileKey) + '.*')
        img = plt.imread(imgFile)
        img = plot_bbox(img, annots)
        plt.imshow(img)
        plt.show()

if __name__=='__main__':
    _ = ICDAR2017('/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Icdar_2017/images/train',
                  '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Icdar_2017/labels/train')
    _.writeToCoCo128()