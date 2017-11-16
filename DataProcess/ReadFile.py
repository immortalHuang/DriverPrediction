# -*- coding: utf-8 -*-
from pyspark.mllib.tree import DecisionTree
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.linalg import Vectors

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("spark://127.0.1.1:7077") \
    .config("spark.driver.memory","1g") \
    .getOrCreate()

sc = spark.sparkContext

OriginTrainData = sc.textFile("../Data/train.csv").map(lambda x:x.split(',')).filter(lambda x:x[0]!='id').map(lambda x:tuple((float(x[1]),Vectors.dense([float(x[i]) for i in range(0,len(x)-1) if i!=1])))).collect()
SparkTrainData = spark.createDataFrame(OriginTrainData,['label','features'])

(trainingData, testData) = SparkTrainData.randomSplit([0.7, 0.3])

model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=5, maxBins=32)

predictions = model.predict(testData.map(lambda x: x.features))

predictions.show()
# lsvc = LinearSVC(maxIter=10, regParam=0.1)
#
# lsvcModel = lsvc.fit(SparkTrainData)
#
# result = lsvcModel.pre
#
# print("Coefficients: " + str(lsvcModel.coefficients))
# print("Intercept: " + str(lsvcModel.intercept))

spark.stop()