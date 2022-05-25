from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator

from pathlib import Path
from typing import Dict
import cv2
import matplotlib.pyplot as plt

from scripts.utils import pdf2images, plot_bbox, resize_bboxes


class ExtractPdfText(object):
    def __init__(self, pdfDir: str):
        pdfDir = Path(pdfDir)
        if not pdfDir.exists():
            raise Exception(f"PDF Dir Doesn't Exists: {pdfDir.as_posix()} !")

        self.pdfFiles = list(pdfDir.iterdir())

        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        self.device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        self.interpreter = PDFPageInterpreter(rsrcmgr, self.device)
        self.chars_coord = dict()  # { {pdfFileName: { PgNo:  [ [x_min, y_min, x_max, y_max, text], ... ], ...}, ...}
        self.word_coord = dict()  # {  {pdfFileName: { PgNo:  [ [x_min, y_min, x_max, y_max, text], ... ], ...}, ...}

    def savePdf2Images(self, imgDir: str = ''):
        for file in self.pdfFiles:
            pdf2images([file.as_posix()], imgDir)

    def parsePDF(self):
        for file in self.pdfFiles:
            self.fp = open(file.as_posix(), 'rb')
            self.pages = PDFPage.get_pages(self.fp)
            textCoordsDict = self.get_text_coordinate(self.pages)
            self.chars_coord[file.stem] = textCoordsDict.get('CharCoords')
            self.word_coord[file.stem] = textCoordsDict.get('WordCoords')

    def get_text_coordinate(self, pgs) -> Dict:
        pgCharsBBoxes = dict()
        pgWordsBBoxes = dict()
        for pgNo, pg in enumerate(pgs):
            chars_coord = list()
            words_coord = list()
            self.interpreter.process_page(pg)
            pgLayout = self.device.get_result()
            for layoutObj in pgLayout:
                if isinstance(layoutObj, LTTextBox):
                    for layoutTextBox in layoutObj:
                        for layoutChar in layoutTextBox:
                            try:
                                bbox_resized = resize_bboxes([[layoutChar.x0, layoutChar.y0, layoutChar.x1, layoutChar.y1,
                                                               layoutChar.get_text()]],
                                                             (pg.cropbox[-1], pg.cropbox[-2]), (1000, 1500))
                                chars_coord.append(bbox_resized[0])
                            except:
                                continue
                        try:
                            words_coord.append([layoutTextBox.x0, layoutTextBox.y0, layoutTextBox.x1, layoutTextBox.y1,
                                                layoutTextBox.get_text()])
                        except:
                            continue
            pgCharsBBoxes[pgNo] = chars_coord
            pgWordsBBoxes[pgNo] = words_coord

        return dict(CharCoords=pgCharsBBoxes, WordCoords=pgWordsBBoxes)

    def draw_annotsOnImages(self, imgDir: str):
        for templateName, imgAnnot in self.chars_coord.items():
            for imgName, annots in imgAnnot.items():
                imgFile = '/'.join([imgDir, templateName, str(imgName)])
                imgFile = imgFile + '.png'
                img = cv2.imread(imgFile)
                img = plot_bbox(img, annots)
                plt.imshow(img)
                plt.show()


if __name__ == "__main__":
    pdf_file_dir = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/PDFs_Resized_Images_Dynamic_Annotations'
    img_dirt = '/home/mansoor/Projects/Craft-Training/dataset/CNIC-Static&Digital/Digital_Anotations'
    extracted_PDF = ExtractPdfText(pdf_file_dir)
    extracted_PDF.parsePDF()
    # extracted_PDF.savePdf2Images(img_dirt)
    extracted_PDF.draw_annotsOnImages(img_dirt)
    print()
