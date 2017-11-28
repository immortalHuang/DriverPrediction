from string import Template

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Spark_decision_tree_classifier").master("local[*]").getOrCreate()

dataList = [
    (0, "female", 32, 1.5, "no", 2, 17, 5, 5),
    (0, "male", 57, 15.0, "yes", 5, 18, 6, 5),
    (0, "female", 32, 15.0, "yes", 1, 12, 1, 4),
    (0, "female", 27, 4.0, "no", 4, 14, 6, 4),
    (0, "male", 37, 10.0, "no", 3, 18, 7, 4),
    (0, "male", 22, 0.75, "no", 2, 17, 6, 3)
]
data = spark.createDataFrame(dataList,["affairs", "gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating"])

data.createOrReplaceTempView("data")

labelWhere = "case when affairs=0 then 0 else cast(1 as double) end as label"
genderWhere = "case when gender='female' then 0 else cast(1 as double) end as gender"
childrenWhere = "case when children='no' then 0 else cast(1 as double) end as children"

dataLabelDF = spark.sql(Template('select ${labelWhere}, ${genderWhere},age,yearsmarried,${childrenWhere},religiousness,education,occupation,rating from data'). \
substitute(labelWhere=labelWhere,genderWhere=genderWhere,childrenWhere=childrenWhere))

featuresArray = ["gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating"]

assembler = VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
vecDF = assembler.transform(dataLabelDF)

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(vecDF)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(vecDF)
(trainingData, testData) = vecDF.randomSplit([0.7, 0.3])

dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",maxBins=100,
                            impurity="entropy",minInfoGain=0.01,minInstancesPerNode=10,seed=123456)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel").setLabels(labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt, labelConverter])

print trainingData.count()
model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "features").show(10, truncate = False)