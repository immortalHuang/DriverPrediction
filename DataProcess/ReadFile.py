# -*- coding: utf-8 -*-
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, DoubleType, StructType, FloatType, Row
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("local[*]") \
    .config("spark.driver.memory","3g") \
    .getOrCreate()

sc = spark.sparkContext

dataFile = sc.textFile("../Data/train.csv").map(lambda e : e.split(',')).map(lambda e:
                                                                             tuple([x if e[0]=='id' else float(x) for x in e ])).collect()

tempScheme = []
for field in dataFile[0]:
    schemeType = StructField(field, FloatType(), True)
    tempScheme.append(schemeType)
schema = StructType(tempScheme)

data = spark.createDataFrame(dataFile[1:],schema)
# data.createOrReplaceTempView("data")

assembler = VectorAssembler().setInputCols(dataFile[0][2:]).setOutputCol("features")
vecDF = assembler.transform(data)

labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(vecDF)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(vecDF)

(trainingData, testData) = vecDF.randomSplit([0.7, 0.3])

dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",maxBins=100,
                            impurity="entropy",minInfoGain=0.01,minInstancesPerNode=10,seed=123456)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel").setLabels(labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt, labelConverter])
model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.select("predictedLabel", "target", "features").show(10, truncate = False)

spark.stop()