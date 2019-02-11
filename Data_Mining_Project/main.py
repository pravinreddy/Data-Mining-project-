import os
import pandas as pd
from pandas import datetime
import matplotlib as mp
import numpy as np
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
#from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore")

""" I have refered this armia model from the "https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/" """

def parser(x):
	return datetime.strptime(x, '%Y-%m')


dates = []

n = 0
y = 0
m = 0

while n<118:
    m = n%12
    if m==0:
        y = y+1
    dates.append(parser("190"+str(y-1)+"-"+str(m+1)))
    n = n+1



# creating the coloumn names for the product distubution traing set
columnNames = ["productId"]
for i  in range(0,118):
    columnNames.append('d'+str(i))


#retreving the data from the
productDistribution = pd.read_csv('product_distribution_training_set.txt', delimiter="\t", names = columnNames)
keyProducts = pd.read_csv('key_product_IDs.txt', delimiter="\t",header = None )


#transposing the coloumns into rows and rows into coloumns
keyProductsTranspose = keyProducts.transpose()
productDistributionTranspose = productDistribution.transpose()


#converting the pandas data frame of key products to the list data structure
keyProductsList = list()

for item in keyProductsTranspose.values[0] :
    keyProductsList.append(item)




trainDays = 117
predictionDays = 29
totalProductSalePredictionList = [0] * predictionDays
eachDayProductPredictionList = list()


file = open("output.txt", "w+")


for col in  productDistributionTranspose :

    X = productDistributionTranspose[col].values[1:]
    d = {'date': pd.to_datetime(dates), 'sales': X}
    series = pd.DataFrame(d, columns=["date", "sales"])

    data = series.sales

    history = [x for x in data]
    predictions = list()
    totalProductSalePrediction = 0

    for day in range(0,predictionDays):
        model = ARIMA(history, order=(7, 1, 0))
        modelFitting = model.fit(disp=0)
        output = modelFitting.forecast()
        yhat = output[0]
        if yhat[0] < 0:
            yhat[0] = 0.0
        predictions.append(int(round(yhat[0])))
        history.append(yhat[0])
        totalProductSalePredictionList[day] = totalProductSalePredictionList[day] + int(round(yhat[0]))


    eachDayProductPredictionList.append(predictions)
    #print("key Poduct", keyProductsList[col])
    file.write(str(keyProductsList[col]))
    for i  in range(len(predictions)):
        file.write(" "+str(predictions[i]))
    file.write("\n")

file.close()

with open("output.txt", "r+") as file:
    readcontent = file.read()

    file.seek(0,0)
    file.write("0")
    for i in range(len(totalProductSalePredictionList)):
        file.write(" "+str(totalProductSalePredictionList[i]))

    file.write("\n")
    file.write(readcontent)
print("********* Done with the predictions ***********")

