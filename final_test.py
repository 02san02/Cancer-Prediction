from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Input the following arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m1", help="Path with name of json and h5 file", type=str)      # Example, "data/model1", not specifying the extensions. Make sure both the files have the same name
parser.add_argument("-m2", help="Path with name of json and h5 file", type=str)      # Example, "data/model2", not specifying the extensions. Make sure both the files have the same name
parser.add_argument("-t", help="Folder path where all the tiles to be tested are present", type=str)

args = parser.parse_args()

m1, m2, t = args.m1, args.m2, args.t

if not m1:
    raise("Model1 path not specified")

if not m2:
    raise("Model2 path not specified")

if not t:
    raise("Test folder path not specified")

# Function used to load the pretained model
def load_model(path=""):
    
    json_file = open(path+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights(path+".h5")
    print("Model loaded successfully")
    
    return loaded_model

def final_evaluate(model1, model2, path="", folder_name=""):
    # path is the directory where the folder_name named folder is present and folder_name has the tiles which are to be evaluated
    
    # model1 => [normal, cancer]
    # model2 => [luad, lusc]
    
    # Generating the generator for loading the test slides
    datagen = ImageDataGenerator(rescale=1/255)
    test = datagen.flow_from_directory(path, target_size=(224, 224), batch_size=1, classes=[folder_name], class_mode=None, shuffle=False)
    
    # Predicting the probability of tile for each model
    test.reset()
    p1 = model1.predict(test, verbose=1, max_queue_size=200, workers=200)
    test.reset()
    p2 = model2.predict(test, verbose=1, max_queue_size=200, workers=200)
    
    # calculating the percentage of each classes
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    for i in range(p1.shape[0]):
        if(p1[i][0] > p1[i][1]):
            c1 += 1
        else:
            c2 += 1
            if(p2[i][0] > p2[i][1]):
                c3 += 1
            else:
                c4 += 1
                
    print("The precentage of cancer is ", c2/(c1+c2))
    print(["luad", "lusc"][c3 < c4], "cancer detected with", c3/(c3+c4) if c3>c4 else c4/(c3+c4), "probability")
    
    return [c1, c2], [c3, c4]

t = t if t[-1]!="/" else t[:-1]

# Loading both the models
model1 = load_model(m1)
model2 = load_model(m2)

path = ""
for i in t.split("/")[:-1]:
    path += i+"/"

# Predicting the final probabilities
print(final_evaluate(model1, model2, path, t.split("/")[-1]))
