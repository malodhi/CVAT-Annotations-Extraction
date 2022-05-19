from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator


class ExtractPdfTextCoord(object):

    def __init__(self, pdfFile: str):
        self.fp = open(pdfFile, 'rb')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        self.device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        self.interpreter = PDFPageInterpreter(rsrcmgr, self.device)
        self.pages = PDFPage.get_pages(self.fp)
        self.chars_coord = dict()  # {PgNo:  [ [x_min, y_min, x_max, y_max, text], ... ], ...}
        self.word_coord = dict()  # {PgNo:  [ [x_min, y_min, x_max, y_max, text], ... ], ...}
        self.get_characters_coordinate()

    def get_characters_coordinate(self):
        for pgNo, pg in enumerate(self.pages):
            chars_coord = list()
            words_coord = list()
            self.interpreter.process_page(pg)
            pgLayout = self.device.get_result()
            for layoutObj in pgLayout:
                if isinstance(layoutObj, LTTextBox):
                    for layoutTextBox in layoutObj:
                        for layoutChar in layoutTextBox:
                            try: chars_coord.append([layoutChar.x0, layoutChar.y0, layoutChar.x1, layoutChar.y1,
                                                     layoutChar.get_text()])
                            except: continue
                        try: words_coord.append([layoutTextBox.x0, layoutTextBox.y0, layoutTextBox.x1, layoutTextBox.y1,
                                                 layoutTextBox.get_text()])
                        except: continue
            self.chars_coord[pgNo] = chars_coord
            self.word_coord[pgNo] = words_coord


if __name__ == "__main__":
    file = '/home/mansoor/Downloads/data (5).pdf08'
    extractions = ExtractPdfTextCoord(file)
