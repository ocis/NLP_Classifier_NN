import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score
import json
import sklearn.preprocessing
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import sys

#to load via inference, use True boolean after Python file name
#loading command line options
if(len(sys.argv) > 1):
    inference = sys.argv[1]
else:
    inference = False

#loading json file
reviewpath = 'yelp_academic_dataset_review.json'
size = 100000
reviews = pd.read_json(reviewpath, nrows=size, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int})
reviews = reviews.drop(['review_id', 'user_id', 'business_id', 'date'], axis=1)

def nnPreProcessing(inputData):
    #filling in blank values for funny, cool, and useful
    inputData.fillna(0)
    #performing scaling and rounding of funny, cool, and useful scores to stars equivalent
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(1,5), copy=False)    
    inputData['funny'] = scaler.fit_transform(inputData[['funny']])
    inputData['cool'] = scaler.fit_transform(inputData[['cool']])
    inputData['useful'] = scaler.fit_transform(inputData[['useful']])
    inputData['funny'] = inputData['funny'].round(0)
    inputData['cool'] = inputData['cool'].round(0)
    inputData['useful'] = inputData['useful'].round(0)
    le = LabelEncoder().fit(inputData['funny'])
    inputData['funny'] = le.transform(inputData['funny'])
    le = LabelEncoder().fit(inputData['cool'])
    inputData['cool'] = le.transform(inputData['cool'])
    le = LabelEncoder().fit(inputData['useful'])
    inputData['useful'] = le.transform(inputData['useful'])

    return inputData

def trainPreprocessing(inputData, inference=False):
    label_cols = ['stars', 'useful', 'funny', 'cool']
    X_train, X_test, y_train, y_test = train_test_split( inputData['text'], inputData[label_cols], test_size = 0.20, random_state = 0)
    X_train, X_validation, y_train, y_validation = train_test_split( inputData['text'], inputData[label_cols], test_size = 0.25, 
                                                                    stratify=inputData['stars'], random_state = 0)
    if(inference):
        testData = pd.concat([X_test, y_test], axis=1)
        testData.reset_index(drop=True, inplace=True)
        testData.to_json('test_data.jsonl')

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def trainNN(trainX, trainY, valX, valY, category='stars'):
    #setting up vars
    trainY = trainY[category]
    valY = valY[category]
    num_epochs = 66
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stop_words = set(stopwords.words('english')) - {"down", "hadn't", "don't", "nor", "no", "not"}
    stop_list = list(stop_words)

    #performing TF-IDF Vectorization and tokenization
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_list, max_features=1000)
    trainX = vectorizer.fit_transform(trainX)
    valX = vectorizer.fit_transform(valX)
    
    trainX = torch.tensor(scipy.sparse.csr_matrix.todense(trainX)).float()
    valX = torch.tensor(scipy.sparse.csr_matrix.todense(valX)).float()

    #convert labels to tensor
    trainY = torch.tensor(trainY.values)
    valY = torch.tensor(valY.values)

    model = nn.Sequential().to(device)
    model.add_module("hidden1", nn.Linear(1000, 500))
    model.add_module("act2", nn.ReLU())
    model.add_module("hidden2", nn.Linear(500, 100))
    model.add_module("act2", nn.ReLU())
    model.add_module("output", nn.Linear(100, 6))
    model.add_module("outact", nn.LogSoftmax(dim=1))

    loss_fn = nn.NLLLoss()

    logps = model(trainX)

    loss = loss_fn(logps, trainY)

    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_losses = []
    val_losses = []
    test_accuracies = []


    for n in range(num_epochs):
       y_pred = model.forward(trainX)
       loss = loss_fn(y_pred, trainY)
       optimizer.zero_grad()
       loss.backward()
       train_loss = loss.item()
       train_losses.append(train_loss)
       optimizer.step()

       with torch.no_grad():
           model.eval()
           log_ps = model(valX)
           test_loss = loss_fn(log_ps, valY)
           val_losses.append(test_loss)
           ps = torch.exp(log_ps)
           top_p, top_class = ps.topk(1, dim=1)
           equals = top_class == valY.view(*top_class.shape)
           test_accuracy = torch.mean(equals.float())
           test_accuracies.append(test_accuracy)

       model.train()
       print(f"Epoch: {n+1}/{num_epochs}.. ",
             f"Training Loss: {train_loss:.3f}.. ",
             f"Test Loss: {test_loss:.3f}.. ",
             f"Test Accuracy: {test_accuracy:.3f}")
    print("Training Complete")
    torch.save(model, 'nnModel.pt')
    return model
       
def evalNN(model, X_test, y_test, category='stars'):
       y_test = y_test[category]
       stop_words = set(stopwords.words('english')) - {"down", "hadn't", "don't", "nor", "no", "not"}
       stop_list = list(stop_words)
       vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_list, max_features=1000)
       X_test = vectorizer.fit_transform(X_test)
       X_test = torch.tensor(scipy.sparse.csr_matrix.todense(X_test)).float()

              #convert labels to tensor
       y_test = torch.tensor(y_test.values)
       loss_fn = nn.NLLLoss()

       model.eval()
       log_ps = model(X_test)

       test_loss = loss_fn(log_ps, y_test)
       ps = torch.exp(log_ps)
       top_p, top_class = ps.topk(1, dim=1)
       equals = top_class == y_test.view(*top_class.shape)
       test_accuracy = torch.mean(equals.float())
       print(f"Test Loss: {test_loss:.3f}.. ",
             f"Test Accuracy: {test_accuracy:.3f}")
       
def runNeuralNetwork(reviews):
    ppReviews = nnPreProcessing(reviews)
    if(inference):
        ppReviews = pd.read_json(reviewpath, nrows=size, lines=True,
                      dtype={'stars':int,'text':str,'useful':int,
                             'funny':int,'cool':int})
        model = torch.load('nnModel.pt')

    X_train, y_train, X_validation, y_validation, X_test, y_test = trainPreprocessing(ppReviews, inference)
    if not inference:
        model = trainNN(X_train, y_train, X_validation, y_validation)
    evalNN(model, X_test, y_test)

runNeuralNetwork(reviews)
