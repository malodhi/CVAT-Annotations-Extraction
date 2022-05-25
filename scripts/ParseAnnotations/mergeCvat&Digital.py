import numpy as np

from CVAT.xmlPascalVoc import PascalVocXml
from Digital.pdfText import ExtractPdfText
from scripts.utils import resize_bboxes, loadmat

from pathlib import Path
import matplotlib.pyplot as plt


class Merge(object):
    def __init__(self):
        self.ImgTemplateDir = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/Original_Images'
        self.ImgTemplateFmt = '.png'
        self.ImgTemplateDir = Path(self.ImgTemplateDir)
        self.allCharCoords = dict()

        self.static_annots = PascalVocXml(
            datasetDir='/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital',
            imgDir='/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/Original_Images',
            gtFile='/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/Original_Static_Annotations.xml')
        self.dynamic_annots = ExtractPdfText(pdfDir='/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/PDFs_Resized_Images_Dynamic_Annotations')
        self.dynamicImgDir = Path('/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/Digital_Anotations')

        self.static_annots()  # imgBBoxes
        self.dynamic_annots()  # chars_coord

        # <----  pick ONLY those images that have both static & dynamic character coordinates ---->
        self.templateNames = list(
            set(self.static_annots.imgBBoxes.keys()).intersection(set(self.dynamic_annots.chars_coord.keys())))
        self.static_annots.imgBBoxes = dict(
            [(key, val) for key, val in self.static_annots.imgBBoxes.items() if key in self.templateNames])
        self.dynamic_annots.chars_coord = dict(
            [(key, val) for key, val in self.dynamic_annots.chars_coord.items() if key in self.templateNames])

    def __len__(self):
        return len(self.templateNames)

    def resize_digital_pdf_annots(self):
        tempDict = dict()

        for templateName, templateAnnots in self.dynamic_annots.chars_coord.items():

            templateImg = self.ImgTemplateDir / str(templateName + self.ImgTemplateFmt)
            origImage = plt.imread(templateImg)

            if len(origImage.shape) == 3:
                height, width, _ = origImage.shape
            else:
                height, width = origImage.shape

            for imgNo, bboxes in templateAnnots.items():
                if not bboxes:
                    continue
                # todo: remove resizing on documents other than current dyanamically annotated cnics:
                # <----  resize bboxes of dynamic coordinates because its images were resized prior to annotation  ---->
                bboxesResized = resize_bboxes(bboxes, (1000, 1500), (height, width)
                                              )
                if templateName not in list(tempDict.keys()):
                    tempDict[templateName] = dict()
                tempDict[templateName][imgNo] = bboxesResized

        self.dynamic_annots.chars_coord = tempDict

    def merge_static_digital_annots(self):

        # todo: resizing may not be necessary if dynamic/digital annotations are done on same image (size) as static:
        self.resize_digital_pdf_annots()

        mergeDict = dict()
        for templateName, templateDigitalAnnots in self.dynamic_annots.chars_coord.items():
            static_chars_bboxes = self.static_annots.imgBBoxes.get(templateName)
            curDigitalImgDir = self.dynamicImgDir / templateName
            for bboxes, imgFile in zip(templateDigitalAnnots.items(), curDigitalImgDir.iterdir()):
                bboxes[1].extend(static_chars_bboxes)
                mergeDict[templateName + '/' + imgFile.name] = np.asarray(bboxes[1])
        self.allCharCoords = mergeDict

    def mapTextCoords2MatFile(self, sampleMatFile: dict):
        sampleMatFile['imnames'] = np.asarray(list(self.allCharCoords.keys())).reshape(1, -1)



if __name__ == '__main__':
    MatFile = loadmat('/home/mansoor/Projects/Craft-Training/dataset/craft-english/gt.mat')
    mergeAnnotations = Merge()
    mergeAnnotations.merge_static_digital_annots()
    mergeAnnotations.mapTextCoords2MatFile(MatFile)

