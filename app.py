from flask import Flask,render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
#import arff as arf # using 'arff' instead of 'arff2pandas' is not avaiable to install over pip
from keras.models import load_model
from keras import Sequential
from keras import layers
from keras.models import Model
from keras.utils import register_keras_serializable
from sklearn.metrics import accuracy_score, precision_score, f1_score
import time

#Initializing the application 
app = Flask(__name__) 

@app.route('/') #Decorate Charater  (Empty Route)

def index(): 
   return render_template ("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    ecg_file_name = "dataset/ecg2.csv"   
    rcds_size = 500  
    while True:
        message = ""   
        ecg_file_name = "dataset/ecg2.csv"  

        dataChunk = pd.read_csv(ecg_file_name)
        dataChunk = dataChunk.sample(frac=.5)
        dataChunk.dropna() # drop null values 
        dataChunk.drop_duplicates() # drop duplicates 

        anomalies, y_preds, y_pred_classes, accuracy, f1, precision = anomalyV2(dataChunk)

        #print("Anomaly ECGs :", np.sum(anomalies.numpy()))
        #print("Normal ECGs :", len(anomalies) - np.sum(anomalies.numpy()))
        #print ("Accuracy : " , accuracy)
        #print ("F1 : " , f1)
        #print ("Precision : " , precision)  
        anamolies = np.sum(anomalies.numpy()) 
        normal = len(anomalies) - np.sum(anomalies.numpy())

        if anamolies > 0 :
           message = f'Anomalies Detected - Anomaly ECGs :  {str(anamolies)} | Normal ECGs : {str(normal)}'
        else :         
           message = "No Anomalies Found"  
        return render_template('index.html',  msg =message)           


# Setting other anomaly classes also to 0
def ecgclass(classId):
   if str(classId) in str(5) or str(classId) in str(4) or str(classId) in str(3) or str(classId) in str(2):
       return 0
   else: 
       return 1


# Model Anomaly
def anomalyV2(dataChunk):    

    X_test = dataChunk.iloc[:,:-1].values
    y_test = dataChunk.iloc[:,-1].values 

    # Normalize the data
    min_val = tf.reduce_min(X_test)
    max_val = tf.reduce_max(X_test)
    X_test = (X_test - min_val) / (max_val - min_val)

    # Generate reconstruction from the model
    X_test_recon = new_Anomaly(X_test)

    # Calculate loss
    loss = tf.keras.losses.mae(X_test_recon, X_test)

    # Apply the threshold to the loss to get the predictions
    #threshold = 0.04
    threshold = np.mean(loss) + np.std(loss) 

    preds = tf.math.less(loss, threshold)

    #anomalies
    anomalies= tf.math.greater(loss, threshold)

    # Convert predictions to a list
    y_preds = preds.numpy().tolist()
    y_pred_classes = np.round(preds).astype(int)  # assuming a binary classification

    accuracy = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)

    return anomalies, y_preds, y_pred_classes, accuracy, f1, precision


def anomalyV3(dataChunk):    

    X_test = dataChunk.iloc[:,:-1].values
    y_test = dataChunk.iloc[:,-1].values 

    # Normalize the data
    min_val = tf.reduce_min(X_test)
    max_val = tf.reduce_max(X_test)
    X_test = (X_test - min_val) / (max_val - min_val)

    # Generate reconstruction from the model
    X_test_recon = new_Anomaly(X_test)

    # Calculate loss
    #loss = tf.keras.losses.mae(X_test_recon, X_test)

    mse = np.mean(np.power(X_test - X_test_recon, 2), axis=1)

    # Apply the threshold to the loss to get the predictions
    threshold = 0.6
    #threshold = np.mean(loss) + np.std(loss) 
    #preds = tf.math.less(loss, threshold)

    #anomalies
    #anomalies= tf.math.greater(loss, threshold)
    anomalies = mse > threshold

    # Convert predictions to a list
    y_preds =  np.argmax(y_test, axis=1)
    y_pred_classes = anomalies.astype(int)  # assuming a binary classification

    accuracy = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)

    return anomalies, y_preds, y_pred_classes, accuracy, f1, precision



# Declaration of Anomaly Class  
@register_keras_serializable()
class DetectAnomaly(Model):
    def __init__(self, trainable=True, dtype='float32'):
        super(DetectAnomaly, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded    

custom_objects = {'DetectAnomaly': DetectAnomaly}
new_Anomaly = load_model("models/anomaly.keras", custom_objects=custom_objects)



if __name__ == '__main__':
    app.run( debug=True)
