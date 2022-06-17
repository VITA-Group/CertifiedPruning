from PIL import Image, ImageDraw, ImageSequence,ImageFont
import io
from urllib.request import urlopen
import time
import imageio
import os
import cv2
import sys

pic_dir = sys.argv[1]
keyword = sys.argv[2]

def compose_gif(dir):
    image_list=[]
    for x in os.listdir(dir):
        if keyword in x and (x[-3:] in ['png','jpg','jpeg']):
            image_list.append(os.path.join(dir,x))
    image_list=sorted(image_list)
    for x in image_list:
        print(x)
    tmp_image_list=[os.path.join('/'.join(x.split('/')[:-1]),'tmp_'+x.split('/')[-1]) for x in image_list]

    for i,x in enumerate(image_list):
        im=cv2.imread(x)
        cv2.putText(im,str(i),(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3)
        cv2.imwrite(tmp_image_list[i],im)

    gif_images = []
    for path in tmp_image_list:
        gif_images.append(imageio.imread(path))
        os.system('rm '+path)
    imageio.mimsave(os.path.join(dir,keyword+".gif"),gif_images,fps=1.5)



compose_gif(pic_dir)
