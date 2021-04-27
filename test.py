from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse

# Input the following arguments
parser = argparse.ArgumentParser()
parser.add_argument("-model", help="Path with name of json and h5 file", type=str)      # Example, "data/model1", not specifying the extensions. Make sure both the files have the same name
parser.add_argument("-t", help="Folder path where all the tiles to be tested are present", type=str)
parser.add_argument("-c", nargs="+", default=["luad", "lusc"])
parser.add_argument("-m", help="Tesing method (positive or average)", type=str, default="positive")

args = parser.parse_args()

model, t, c, m = args.model, args.t, args.c, args.m

if not model:
    raise("Model path not specified")

if not t:
    raise("Test folder path not specified")

m = m.lower()

# Function used to load the pretained model
def load_model(path=""):
    
    json_file = open(path+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights(path+".h5")
    print("Model loaded successfully")
    
    return loaded_model

# Function used to evaluate a whole-images 
def evaluate(model, path="data/", folder_name="", classes=["luad", "lusc"], method="positive"):
    # path is the directory where the folder_name named folder is present and folder_name has the tiles which are to be evaluated

    datagen = ImageDataGenerator(rescale=1/255)
    test = datagen.flow_from_directory(path, target_size=(224, 224), batch_size=1, classes=[folder_name], class_mode=None, shuffle=False)
    p = model.predict(test, verbose=1, max_queue_size=200, workers=200)
    
    # Calculating the final probability using positively classfied tiles method
    if(method=="positive"):
        counts = [0 for i in range(len(classes))]
        for i in p:
            counts[np.where(i == max(i))[0][0]] += 1
    
        counts = np.array(counts)
        counts = counts/test.n
        print(str(classes[np.where(counts == max(counts))[0][0]]).upper() + " class detected.")
        return counts
        
    # Calculating the final probability by taking the average of the outputted probabilities
    elif(method=="average"):
        p = np.sum(p, axis=0)
        p = p/test.n
        print(str(classes[np.where(p == max(p))[0][0]]).upper() + " class detected.")
        return p

t = t if t[-1]!="/" else t[:-1]

# Loading both the models
model = load_model(model)

path = ""
for i in t.split("/")[:-1]:
    path += i+"/"

# Predicting the final probabilities
print(evaluate(model, path, t.split("/")[-1], c, m))