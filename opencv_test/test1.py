# -*-coding:utf-8-*-

import pytesseract
from PIL import  Image

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

def recognize_text(img):
    '''数字识别
    :param img: 切割后的数字
    :return: 返回字符串
    '''
    textImage = Image.open(img)
    try:
        text = pytesseract.image_to_string(textImage,lang="chi_sim",config='--psm 6')
        print(text.split(" "))
        # return text
        print("This OK:%s" % text)
    except BaseException as e:
        print(e)


if __name__ == "__main__":
    recognize_text('9.png')