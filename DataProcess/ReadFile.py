import os

import pandas as pd
from pyspark.sql import SparkSession
from sklearn import svm
#
OriginTrainData = pd.read_csv("../Data/train.csv")
OriginTestData = pd.read_csv("../Data/test.csv")

TrainDataY = OriginTrainData.pop('target')
OriginTrainData.insert(0,'target',TrainDataY)


#
# clf = svm.SVC()
#
# clf.fit(TrainDataX,TrainDataY)
#
# result = clf.predict(OriginTestData)
#
# print result.head()

from pyspark.ml.classification import LinearSVC

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .getOrCreate() \
    .master("spark://127.0.1.1:7077")


lsvc = LinearSVC(maxIter=10, regParam=0.1)

lsvcModel = lsvc.fit(OriginTrainData)

print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

spark.stop()