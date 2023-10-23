from PIL import Image
from PIL.TiffTags import TAGS

# (220, 140, 0) is the RGB value of a building pixel in GTC files.
# 6 represents a building pixel in GTL files.
# (224, 224, 224) is a background pixel in GTC files.
# 2 represents a background pixel in GTL files.
# (96, 96, 96) is an uncertain building (i.e. part of it is cut off at the border of the image) in GTC files.
# 65 represents an uncertain building in GTL files.

# See https://www.topcoder.com/challenges/db36b53a-c2f3-4899-9698-13e96148ffcd for more information.

with Image.open('Datasets/Urban_3D_Challenge/01-Provisional_Train/GT/JAX_Tile_004_GTL.tif') as img:
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
    print(len(list(img.getdata())))

print(meta_dict.keys())