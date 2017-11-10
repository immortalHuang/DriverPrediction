# -*- coding: utf-8 -*-
import os

import pandas as pd
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, DoubleType
from sklearn import svm
#



#
# clf = svm.SVC()
#
# clf.fit(TrainDataX,TrainDataY)
#
# result = clf.predict(OriginTestData)
#
# print result.head()

from pyspark.ml.classification import LinearSVC
from pyspark.sql import SQLContext


spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("local[4]") \
    .getOrCreate()

sqlContext = SQLContext(spark)

sc = spark.sparkContext


DataStruct = StructType([StructField("label", DoubleType(), True),StructField("features", VectorUDT(), True)])
TrainData = spark.read.csv(
    "../Data/train.csv", header=True, mode="DROPMALFORMED"
)


TrainData.withColumn('id2',TrainData.id+1)


OriginTestData = sc.textFile("../Data/test.csv")
TestData = spark.read.csv(
    "../Data/test.csv", header=True, mode="DROPMALFORMED"
)


# lsvc = LinearSVC(maxIter=10, regParam=0.1)
#
# lsvcModel = lsvc.fit(TrainData)
#
# print("Coefficients: " + str(lsvcModel.coefficients))
# print("Intercept: " + str(lsvcModel.intercept))

spark.stop()