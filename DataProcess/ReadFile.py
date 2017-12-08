# -*- coding: utf-8 -*-

from pyspark.ml.feature import VectorAssembler, HashingTF, Tokenizer, IDF
from pyspark.sql import SparkSession



spark = SparkSession \
    .builder \
    .appName("linearSVC") \
    .master("local[*]") \
    .config("spark.driver.memory","3g") \
    .getOrCreate()

sc = spark.sparkContext
#.map(lambda e:(e,1)).reduceByKey(lambda a,b:a+b).map( lambda e:e.split('\t')).map(lambda e : re.findall('[0-9a-zA-Z]+',e[1])) \
#.sortBy(lambda e:e[1],ascending=False),['train_id','name','item_condition_id','category_name','brand_name','price','shipping','item_description']

data = sc.textFile("../Data/train.tsv").map( lambda e:e.split('\t'))

name_tokenizer = Tokenizer().setInputCol('name').setOutputCol('tokenizer_name')

dataTable = spark.createDataFrame(data.filter(lambda line:line[0] != 'train_id'),data.first())

tokenizer_name_dataTable = name_tokenizer.transform(dataTable)

hashingTF = HashingTF(numFeatures=10000,inputCol="tokenizer_name",outputCol="hash_name")
nameTF = hashingTF.transform(tokenizer_name_dataTable)
idf = IDF().setInputCol('hash_name').setOutputCol('idf_name')
idfModel = idf.fit(nameTF)
name_rescaledData = idfModel.transform(nameTF)


category_name_tokenizer = Tokenizer().setInputCol('category_name').setOutputCol('tokenizer_category_name')
tokenizer_category_name_dataTable = category_name_tokenizer.transform(name_rescaledData)
category_name_hashingTF = HashingTF(numFeatures=10000,inputCol="tokenizer_category_name",outputCol="hash_category_name")
category_name_TF = category_name_hashingTF.transform(tokenizer_category_name_dataTable)
category_name_idf = IDF().setInputCol('hash_category_name').setOutputCol('idf_category_name')
category_name_idfModel = category_name_idf.fit(category_name_TF)
category_name_rescaledData = category_name_idfModel.transform(category_name_TF)


brand_name_tokenizer = Tokenizer().setInputCol('brand_name').setOutputCol('tokenizer_brand_name')
tokenizer_brand_name_dataTable = brand_name_tokenizer.transform(category_name_rescaledData)
brand_name_hashingTF = HashingTF(numFeatures=10000,inputCol="tokenizer_brand_name",outputCol="hash_brand_name")
brand_name_TF = brand_name_hashingTF.transform(tokenizer_brand_name_dataTable)
brand_name_idf = IDF().setInputCol('hash_brand_name').setOutputCol('idf_brand_name')
brand_name_idfModel = brand_name_idf.fit(brand_name_TF)
brand_name_rescaledData = brand_name_idfModel.transform(brand_name_TF)


item_description_tokenizer = Tokenizer().setInputCol('item_description').setOutputCol('tokenizer_item_description')
tokenizer_item_description_dataTable = item_description_tokenizer.transform(brand_name_rescaledData)
item_description_hashingTF = HashingTF(numFeatures=10000,inputCol="tokenizer_item_description",outputCol="hash_item_description")
item_description_TF = item_description_hashingTF.transform(tokenizer_item_description_dataTable)
item_description_idf = IDF().setInputCol('hash_item_description').setOutputCol('idf_item_description')
item_description_idfModel = item_description_idf.fit(item_description_TF)
rescaledData = item_description_idfModel.transform(item_description_TF)

idfData = rescaledData.select('train_id','item_condition_id','price','shipping','idf_name','idf_category_name','idf_item_description','idf_brand_name')


assembler = VectorAssembler().setInputCols(['item_condition_id','shipping','idf_name','idf_category_name','idf_item_description','idf_brand_name'] \
                                           ).setOutputCol("features")

idfData.printSchema()
# vecDF = assembler.transform(idfData)
# labelIndexer = StringIndexer(inputCol="price", outputCol="indexedLabel").fit(vecDF)
# featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(vecDF)
# trainingData = vecDF
#
# dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",maxBins=100,
#                             impurity="entropy",minInfoGain=0.01,minInstancesPerNode=10,seed=123456)
#
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel").setLabels(labelIndexer.labels)
#
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt, labelConverter])
#
# model = pipeline.fit(trainingData)

# testFile = sc.textFile("../Data/test.csv").map(lambda e : e.split(',')).map(lambda e:
#                                                                              tuple([x if e[0]=='id' else float(x) for x in e ])).collect()
#
# tempScheme = []
# for field in dataFile[0]:
#     schemeType = StructField(field, FloatType(), True)
#     tempScheme.append(schemeType)
# schema = StructType(tempScheme)
#
# testSchema = StructType(tempScheme[0:1] + tempScheme[2:])
#
# data = spark.createDataFrame(dataFile[1:],schema)
# tdata = spark.createDataFrame(testFile[1:],testSchema)
# # data.createOrReplaceTempView("data")
#
# assembler = VectorAssembler().setInputCols(dataFile[0][2:]).setOutputCol("features")
# vecDF = assembler.transform(data)
# testData = VectorAssembler().setInputCols(testFile[0][1:]).setOutputCol("features").transform(tdata)
#
# labelIndexer = StringIndexer(inputCol="target", outputCol="indexedLabel").fit(vecDF)
# featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(vecDF)
#
# trainingData = vecDF
#
# dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",maxBins=100,
#                             impurity="entropy",minInfoGain=0.01,minInstancesPerNode=10,seed=123456)
#
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel").setLabels(labelIndexer.labels)
#
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt, labelConverter])
# model = pipeline.fit(trainingData)
#
# predictions = model.transform(testData)
#
# predictions.select("predictedLabel", "target", "features").show(10, truncate = False)

spark.stop()