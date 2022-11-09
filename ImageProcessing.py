import os
import numpy as np
from keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from sklearn.neural_network import MLPClassifier
def features_extraction():
    try:
        data = []
        labels = []
        classes = 43
        cur_path = os.getcwd()

        #Retrieving the images and their labels 
        for i in range(classes):
            path = os.path.join(cur_path,'train',str(i))
            #print(path)
            images = os.listdir(path)

            for a in images[:200]:
                #print("path2="+path + '\\'+ a)
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                #sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
         
                    
             

        #Converting lists into numpy arrays
        data = np.array(data)
        data = data.reshape(len(data), -1)
        labels = np.array(labels)


        print("[INFO] Image Processing completed")

        return data,labels
    except Exception as e:
        print(e)
if __name__ == '__main__':
    features_extraction()
    