# -*- coding: utf-8 -*-
from pyspark.ml.linalg import VectorUDT
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, DoubleType
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def process(s):
    words = s.split(',')
    return (words[1],Vectors.dense([float(words[2]),float(words[3])]))

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("spark://127.0.1.1:7077") \
    .getOrCreate()

sqlContext = SQLContext(spark)

sc = spark.sparkContext

OriginTrainData = sc.textFile("file:///mnt/f/workspace/DriverPrediction/Data/train.csv").map(lambda x:x.split(',')).filter(lambda x:x[0]!='id').map(lambda x:tuple((float(x[1]),Vectors.dense([float(x[2]),float(x[3])])))).collect()
SparkTrainData = spark.createDataFrame(OriginTrainData,['label','features'])



lsvc = LinearSVC(maxIter=10, regParam=0.1)

lsvcModel = lsvc.fit(SparkTrainData)

print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

# spark.stop()