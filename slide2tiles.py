import numpy as np
import slideio
import matplotlib.pyplot as plt
import os
import argparse

# Input the following arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Path of the input slide image", type=str)
parser.add_argument("-o", help="Path where all the tiles should be saved", type=str)
parser.add_argument("-m", help="Tiling method (rgb or grey)", type=str, default="rgb")

args = parser.parse_args()

i, o, m = args.i, args.o, args.m

m = m.lower()

if not i:
    raise("Input path not specified.")

if not o:
    raise("Output path not specified.")

# Function to convert a particular slide into tiles 
def make_tiles(input_path="", output_path="", method="rgb"):
    
    if(len(os.listdir(output_path)) != 0):
        raise("Output folder not empty")
    
    # Loading the slide image
    slide = slideio.open_slide(input_path, "SVS")
    scene = slide.get_scene(0)
    image = scene.read_block()
    shape = image.shape
    
    # Splitting into tiles
    count = 0
    for i in range(0, image.shape[1]-224, 224):
        for j in range(0, image.shape[0]-224, 224):
            data = np.array(scene.read_block((i, j, 224, 224)))
            if(method=="rgb"):
                temp = data
                temp = temp > 220
                if(not np.sum(np.sum(np.sum(temp, axis=-1), axis=-1), axis=-1) > (224*224*3)//2):
                    plt.imsave(output_path+str(count)+".png", data)
                    count += 1
            elif(method=="grey"):
                temp = data
                temp = np.sum(temp, axis=-1)/3
                temp = temp > 220
                if(not np.sum(np.sum(temp, axis=-1), axis=-1) > (224*224)//2):
                    plt.imsave(output_path+str(count)+".png", data)
                    count += 1
            else:
                raise("Method not valid.")
                return
            
    print("Conversion to tiles done successfully")

make_tiles(input_path=i, output_path=o, method=m)