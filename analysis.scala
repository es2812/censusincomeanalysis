/* 
 * Experimentos de clasificación para el dataset Census Income Dataset de UCI.
 *
 * Autor: Esther Cuervo Fernández
 *        25-03-2019
 *
 */

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType,IntegerType}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

/*
 *          Algunas definiciones
 */
val PATH="./datos/"
val DATA="adult.data"
val TEST="adult.test"

val ATTR= Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","cap-gain","cap-loss", "hours-week", "native-country", "label")
val CONTATTR = Array("age","fnlwgt","education-num","cap-gain","cap-loss","hours-week")
val CATATTR = ATTR.diff(CONTATTR)

/*
 *          Leyendo datos
 */

val raw = sc.textFile(PATH+DATA)
val parsed = raw.filter(_.nonEmpty).map(line=>line.split(", "))

// los valores faltantes se representan con "?", lo podemos cambiar por null para poder leer los datos a un DF
// sin embargo para la primera aproximación los eliminamos
print("ORIGINAL:"+parsed.count)
val dataRDD = parsed.filter(x=>{!(x contains("?"))})
print("WITHOUT NULLS:"+dataRDD.count)

val dataRDDNumeric = dataRDD.map(a=>Array(a(0).toDouble,a(1),a(2).toDouble,a(3),a(4).toDouble,a(5),a(6),a(7),a(8),a(9),a(10).toDouble,a(11).toDouble,a(12).toDouble,a(13),a(14)))

//creamos un Schema para el dataframe, con los tipos apropiados
val adultSchema =StructType(Array( 
  StructField("age",DoubleType,true),
  StructField("workclass",StringType,true),
  StructField("fnlwgt",DoubleType,true), 
  StructField("education",StringType,true),
  StructField("education-num",DoubleType,true),
  StructField("marital-status",StringType,true), 
  StructField("occupation",StringType,true), 
  StructField("relationship",StringType,true), 
  StructField("race",StringType,true), 
  StructField("sex",StringType,true),
  StructField("cap-gain",DoubleType,true), 
  StructField("cap-loss",DoubleType,true), 
  StructField("hours-week",DoubleType,true), 
  StructField("native-country",StringType,true), 
  StructField("label",StringType,true) 
)) 

val dataDF = spark.createDataFrame(dataRDDNumeric.map(Row.fromSeq(_)),adultSchema)

import org.apache.spark.ml.feature.{StringIndexer,StringIndexerModel}

/*
 *  Transformamos el DataFrame en uno con columna label y features, para lo cual debemos convertir los strings en doubles. 
 *  Utilizamos un StringIndexer.
 */

def indexStringColumns(df:DataFrame, cols:Array[String]):DataFrame = {
  var newdf = df
  for(col <- cols) {
    val si = new
    StringIndexer().setInputCol(col).setOutputCol(col+"-numeric")
    val sm:StringIndexerModel = si.fit(newdf)
    newdf = sm.transform(newdf).drop(col)
    newdf = newdf.withColumnRenamed(col+"-numeric", col)
  }
  newdf
}

val numericDF = indexStringColumns(dataDF,CATATTR)

import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
assembler.setOutputCol("features")
assembler.setInputCols(numericDF.columns.diff(Array("label")))

val mlDF = assembler.transform(numericDF).select("features","label")

/*
 * Creamos un árbol de decisión con parámetros prefijados
 *  IMPORTANTE: El valor maxBins debe valer el menos como el mayor número de distintos valores categoricos
 */

import org.apache.spark.ml.classification.DecisionTreeClassifier
val DT=new DecisionTreeClassifier()
val impureza = "entropy"
val maxBins = 41

DT.setImpurity(impureza)
DT.setMaxBins(maxBins)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder

val meval = new MulticlassClassificationEvaluator()
meval.setMetricName("accuracy")

val crossval = new CrossValidator().setEstimator(DT).setEvaluator(meval)

val paramGrid = new ParamGridBuilder().addGrid(DT.maxDepth, Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20)).build()
crossval.setEstimatorParamMaps(paramGrid)
crossval.setNumFolds(5)

val cvModel = crossval.fit(mlDF)

val DTmodel = cvModel.bestModel
DTmodel.extractParamMap()
