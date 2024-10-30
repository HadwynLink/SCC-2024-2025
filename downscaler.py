from PIL import Image
import cv2
import os

def downscale_image(image_path, new_width, new_height):
    with Image.open(image_path) as img:
        img = img.resize((new_width, new_height), Image.LANCZOS) 
        img.save(image_path)


for filename in os.listdir("data/SPARK/train+val"):
    filedir = os.path.join("data/SPARK/train+val", filename)
    downscale_image(filedir, 256, 256)
    print(filename)
print("done")