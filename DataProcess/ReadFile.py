# -*- coding: utf-8 -*-
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("spark://127.0.1.1:7077") \
    .config("spark.driver.memory","1g") \
    .getOrCreate()

sc = spark.sparkContext


OriginTrainData = sc.textFile("../Data/train.csv").map(lambda x:x.split(',')).filter(lambda x:x[0]!='id').map(lambda x:LabeledPoint(x[1],[float(x[i]) for i in range(0,len(x)-1) if i!=1 and i!=0]))
# SparkTrainData = spark.createDataFrame(OriginTrainData,['label','features'])

(trainingData, testData) = OriginTrainData.randomSplit([0.7, 0.3])

model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={0:2},
                                    impurity='variance', maxDepth=5, maxBins=32)

predictions = model.predict(testData.map(lambda x:x.features))

print predictions.collect()
# lsvc = LinearSVC(maxIter=10, regParam=0.1)
#
# lsvcModel = lsvc.fit(SparkTrainData)
#
# result = lsvcModel.pre
#
# print("Coefficients: " + str(lsvcModel.coefficients))
# print("Intercept: " + str(lsvcModel.intercept))

spark.stop()