import numpy as np
import os
import shutil
import argparse

# Input the following arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Path where the train folder is present", type=str)
parser.add_argument("-p", help="The train-validation-test split percentage", type=int, default=20)      # Combined percentage of validation and test
parser.add_argument("-c", nargs="+", default=["luad", "lusc"])

args = parser.parse_args()

i, p, c = args.i, args.p, args.c

if not i:
    raise("Input path not specified.")

if i[-1] != "/":
    i += "/"

# Function to split the dataset into train, validation and test
def dataset_split(path="/", classes=["luad", "lusc"], sizes=[], split_percentage=10.0):
    
    try:
        for c in classes:
            os.makedirs(path+"test/"+c)
            os.makedirs(path+"validation/"+c)
    except:
        pass
    
    # Size of each split for each class    
    val_test_sizes = np.array(sizes)*(split_percentage/(2*100))
    
    # Randomly picking the slide tiles to move to their respective sets (This will generate an index value)
    validation = []
    test = []
    for i in range(len(classes)):
        l1 = []
        while(len(l1) < int(val_test_sizes[i])):
            num = np.random.randint(0, sizes[i])
            if num not in l1:
                l1.append(num)
        test.append(l1)
        
        l2 = []
        while(len(l2) < int(val_test_sizes[i])):
            num = np.random.randint(0, sizes[i])
            if num not in l2 and num not in l1:
                l2.append(num)
        validation.append(l2)
    

    # Getting the names of slide images of each classes
    class_images = []
    for c in classes:
        class_images.append(os.listdir(path+"train/"+c))
    
    # Moving the tiles to their respective sets
    for i in range(len(classes)):
        c = classes[i]
        for t in validation[i]:
            shutil.move(path+"train/"+c+"/"+class_images[i][t], path+"validation/"+c+"/"+class_images[i][t])
        
        for t in test[i]:
            shutil.move(path+"train/"+c+"/"+class_images[i][t], path+"test/"+c+"/"+class_images[i][t])
        
    print("Split done successfully")
            
    return

sizes = []
for clas in os.listdir(i+"train/"):
    sizes.append(len(os.listdir(i+"train/"+clas)))

dataset_split(path=i, classes=c, sizes=sizes, split_percentage=p)