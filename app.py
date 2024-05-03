import requests
from flask import Flask,render_template, jsonify, request
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
from keras import layers
from keras.models import Model
from keras.utils import register_keras_serializable
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
#import time
import matplotlib.pyplot as plt
import time

#Initializing the application 
app = Flask(__name__) 

@app.route('/') #Decorate Charater  (Empty Route)

def index(): 
   return render_template ("index.html")

@app.route('/predict', methods = ["POST"])
def predict():   
    data = request.get_json() 
    frctn = data['frctn'] # reading the fraction of data as a query string 
    
    api_url = f"https://mlcw2-api.onrender.com/GetData?Fraction={frctn}"
     
    result = ""           
    response = requests.get(api_url)
    dataChunk = pd.DataFrame(response.json())
    
    anomaly, normal, best_model = func_DataPreProcess(dataChunk)
 

    result = f'Anomalies Detected - Anomaly ECGs :  {str(len(anomaly))} | Normal ECGs : {str(len(normal))}'   
    best_model = 'Predicted By : ' +  best_model + ' Model'
    if len(anomaly) > 0:        
        color = 'red'
    else : 
        color = 'green'
    
    return jsonify({'message': result, 'color': color, 'bestmodel' : best_model })
 

# Declaration of Auto Encoder Anomaly Class  
@tf.keras.utils.register_keras_serializable()
class AutoEncoder_DetectAnomaly(Model):
    def __init__(self, trainable=True, dtype='float32'):
        super(AutoEncoder_DetectAnomaly, self).__init__()
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
    
custom_objects = {'DetectAnomaly': AutoEncoder_DetectAnomaly}
new_AutoEncoder = load_model("models/autoencoder.keras", custom_objects=custom_objects)


# Multi class classification to Binary Class
def func_EcgGlass(classId):
    if str(classId) in str(5) or str(classId) in str(4) or str(classId) in str(3) or str(classId) in str(2):
        return 0
    else: 
        return 1

   

#Feed Forwarding Network (FFN) Model Prediction 
def func_FFN (X_test, y_test):
    # Load the FFN Model
    new_ffn = load_model("models/ffn.keras") 
    
    preds_prob = new_ffn.predict(X_test)    
    preds = (preds_prob > 0.5).astype("int32")
    # Filter the anomalies
    anomalies = X_test[preds.reshape(-1) == 1]
    normal_data = X_test[preds.reshape(-1) == 0]
    
    test_preds = (new_ffn.predict(X_test) > 0.5).astype("int32")
    test_labels = y_test.astype(bool)

    accuracy = accuracy_score(test_labels, test_preds) # Accuracy (FFN) 
    f1 = f1_score(test_labels, test_preds)             # F1 (FNN)  
    precision = precision_score(test_labels, test_preds) # Precision (FFN) 
    
    return anomalies,normal_data, accuracy, f1, precision



# Auto Encoder Model Prediction 
def func_AutoEncoder(X_test, y_test):
       
    X_test_recon = new_AutoEncoder(X_test) #Generate reconstruction from the model
    loss = tf.keras.losses.mae(X_test_recon, X_test) # Calculate loss
    
    #Apply the threshold to the loss to get the predictions    
    threshold = np.mean(loss) + np.std(loss) 

    #Normal 
    normal_preds = tf.math.less(loss, threshold)

    #Anomalies
    anomaly_preds= tf.math.greater(loss, threshold)
        
    #y_preds = normal_preds.numpy().tolist() # Predictions to a list
    y_pred_classes = np.round(normal_preds).astype(int) # As Binary

    accuracy = accuracy_score(y_test, y_pred_classes)   # Accuracy (Auto encoder) 
    f1 = f1_score(y_test, y_pred_classes)               # F1 Score (Auto encoder) 
    precision = precision_score(y_test, y_pred_classes) # Precision (Auto encoder)  
    return anomaly_preds, accuracy, f1, precision


# LSTM Model Prediction 
def func_LSTM(X_test, y_test):
    # Load the LSTM model
    new_lstm = load_model("models/lstm.keras")
    
    # Make predictions on the test data
    preds_prob = new_lstm.predict(X_test)
    preds = (preds_prob > 0.5).astype("int32")
    
    # Filter anomalies and normal data
    anomalies = X_test[preds.reshape(-1) == 1]
    normal_data = X_test[preds.reshape(-1) == 0]
    
    # Convert test labels to boolean for comparison
    test_preds = (new_lstm.predict(X_test) > 0.5).astype("int32")
    test_labels = y_test.astype(bool)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, test_preds)      # Accuracy (LSTM)
    f1 = f1_score(test_labels, test_preds)                  # F1 Score (LSTM)  
    precision = precision_score(test_labels, test_preds)    # Precision (LSTM)  
    return anomalies, normal_data, accuracy, f1, precision


# RNN Model Prediction 
def func_RNN (X_test, y_test):
    new_RNN = load_model("models/rnn.keras")
       
    preds_prob = new_RNN.predict(X_test)    
    preds = np.round(preds_prob)
    # Filter the anomalies
    anomalies = X_test[preds.reshape(-1) == 1]
    normal_data = X_test[preds.reshape(-1) == 0]
    
    test_preds = preds
    test_labels = y_test.astype(bool)
    
    accuracy = accuracy_score(test_labels, test_preds)       # Accuracy (RNN) 
    f1 = f1_score(test_labels, test_preds)                   # F1 (RNN)  
    precision = precision_score(test_labels, test_preds)     # Precision (RNN)
    return anomalies,normal_data, accuracy, f1, precision



# Data Pre-Processing Stage
def func_DataPreProcess (dataChunk) :    
    
    # multi class classification to binary classification 
    dataChunk['140'] = dataChunk['140'].apply(func_EcgGlass)     # Apply the ecg_class function to the column
    dataChunk.dropna() # drop null values 
    dataChunk.drop_duplicates() # drop duplicates 
    
    X_test = dataChunk.iloc[:,:-1].values  # Data
    y_test = dataChunk.iloc[:,-1].values   # Target
            
    # Normalize the data
    min_val = tf.reduce_min(X_test)
    max_val = tf.reduce_max(X_test)
    X_test = (X_test - min_val) / (max_val - min_val)    
    
     
    # 1. Passing the data to FNN  
    fnn_anomalies, fnn_normal, fnn_accuracy, fnn_f1, fnn_precision = func_FFN(X_test,y_test)     
 
    # 2. Passing the data to Auto Encoder      
    aEnc_anomalies, aEnc_accuracy, aEnc_f1, aEnc_precision = func_AutoEncoder(X_test,y_test)  
    aEnc_normal = len(aEnc_anomalies) - np.sum(aEnc_anomalies.numpy()) 

    # 3. Passing the data to LSTM     
    lstm_anomalies, lstm_normal, lstm_accuracy, lstm_f1, lstm_precision = func_LSTM(X_test,y_test)  
            
    # 4. Passing the data to RNN
    rnn_anomalies, rnn_normal, rnn_accuracy, rnn_f1, rnn_precision = func_RNN(X_test,y_test)


    fnn_metrics = func_StoreMetrics("FFN", fnn_accuracy, fnn_f1, fnn_precision) # Store FNN Matrices 
    aenc_metrics = func_StoreMetrics("AutoEncoder", aEnc_accuracy, aEnc_f1, aEnc_precision) # Store Auto Encoder Matrices 
    lstm_metrics = func_StoreMetrics("LSTM", lstm_accuracy, lstm_f1, lstm_precision) # Store LSTM Matrices 
    rnn_metrics = func_StoreMetrics("RNN", rnn_accuracy, rnn_f1, rnn_precision)   # Store RNN Matrices 
    
    # Store the metrics in to a list
    all_metrics = [fnn_metrics,aenc_metrics, lstm_metrics, rnn_metrics]  #fnn_metrics, aenc_metrics, lstm_metrics, rnn_metrics
    # Select the best model
    best_model = func_GetBestModel(all_metrics)
    

    if (best_model == 'FFN'):
       return fnn_anomalies, fnn_normal, best_model
    if (best_model == "AutoEncoder"):
       return aEnc_anomalies, aEnc_normal, best_model
    if (best_model == "LSTM"):
       return lstm_anomalies, lstm_normal, best_model
    if (best_model == "RNN"):
       return rnn_anomalies, rnn_normal,  best_model
    else:
       raise ValueError("No best model was selected. Please check the metrics and selection criteria.")
    
      

# Structure to store the metrics for each model
def func_StoreMetrics(model_name, accuracy, f1, precision):
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision
    }

#function to compare the metrics
def func_GetBestModel(metrics_list):
    
    weights = {'accuracy': 0.4, 'f1': 0.3, 'precision': 0.3}    
    best_score = 0 # Initialize the best score and model name
    best_model = None
    
    for metrics in metrics_list:
        # Calculate weighted score
        weighted_score = (weights['accuracy'] * metrics['accuracy'] +
                          weights['f1'] * metrics['f1'] +
                          weights['precision'] * metrics['precision'])

        # Update the best model if the current model's score is higher
        if weighted_score > best_score:
            best_score = weighted_score
            best_model = metrics['model_name']

    return best_model





# Create a Confusion Matrix
def func_confusion_matrix (test_labels, test_preds, model_name, cMap):
    conf_matrix = confusion_matrix(test_labels, test_preds)
    # Plotting the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cMap,xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])   
    plt.title('Confusion Matrix -' + model_name)
    imgId = 'cm_' + str(int(time.time())) + '.png'
    imgUrl = 'static/images/' + imgId
    plt.savefig(imgUrl) 
    return imgUrl

if __name__ == '__main__':
    app.run( debug=True)
