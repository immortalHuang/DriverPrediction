# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("local[*]") \
    .getOrCreate()

sqlContext = SQLContext(spark)

sc = spark.sparkContext

OriginTrainData = sc.textFile("../Data/train.csv").map(lambda x:x.split(',')).filter(lambda x:x[0]!='id').map(lambda x:tuple((float(x[1]),Vectors.dense([float(x[i]) for i in range(0,len(x)-1) if i!=1])))).collect()
SparkTrainData = spark.createDataFrame(OriginTrainData,['label','features'])


lsvc = LinearSVC(maxIter=10, regParam=0.1)

lsvcModel = lsvc.fit(SparkTrainData)

print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

# spark.stop()