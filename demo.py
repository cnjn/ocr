from model import OcrHandle
from PIL import Image
import logging
from config import dbnet_max_size

logger = logging.getLogger('logger.' + __name__)
ocrhandle = OcrHandle()

short_size = 800
img = Image.open("test.png")

try:
    if hasattr(img, '_getexif') and img._getexif() is not None:
        orientation = 274
        exif = dict(img._getexif().items())
        if orientation not in exif:
            exif[orientation] = 0
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
except Exception as ex:
    logger.error(str(ex), exc_info=True)
    exit(0)

img = img.convert("RGB")
res = []
do_det = True
if short_size < 64:
    res.append("短边尺寸过小，请调整短边尺寸")
    do_det = False

short_size = 32 * (short_size // 32)

img_w, img_h = img.size
if max(img_w, img_h) * (short_size * 1.0 / min(img_w, img_h)) > dbnet_max_size:
    res.append("图片resize后长边过长，请调整短边尺寸")
    do_det = False

if do_det:
    res = ocrhandle.text_predict(img, short_size)
    if res:
        for i in res:
            print(i[1][i[1].find("、")+1:])
    else:
        print("错误!请调整短边尺寸")
