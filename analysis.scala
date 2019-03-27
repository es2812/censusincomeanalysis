/* 
 * Experimentos de clasificación para el dataset Census Income Dataset de UCI.
 *
 * Autor: Esther Cuervo Fernández
 *        25-03-2019
 *
 */

import org.apache.spark.rdd.RDD

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType,IntegerType}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.{StringIndexer,StringIndexerModel}
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

/*
 *          Algunas definiciones
 */
val PATH="./datos/"
val DATA="adult.data"

val ATTR= Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss", "hours-week", "native-country", "label")
val CONTATTR = Array("age","fnlwgt","education-num","capital-gain","capital-loss","hours-week")
val CATATTR = ATTR.diff(CONTATTR)

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
  StructField("capital-gain",DoubleType,true), 
  StructField("capital-loss",DoubleType,true), 
  StructField("hours-week",DoubleType,true), 
  StructField("native-country",StringType,true), 
  StructField("label",StringType,true) 
)) 

/*
 *          Algunas funciones
 */

def getStringRDD(path:String, file:String):RDD[Array[String]] = {
  // los valores faltantes se representan con "?", lo podemos cambiar por null para poder leer los datos a un DF
  // sin embargo para la primera aproximación los eliminamos
  sc.textFile(path+file).filter(_.nonEmpty).map(line=>line.split(", ")).filter(x=>{!(x.contains("?"))})
}

def selectAttributes(df:DataFrame, attrs:Array[String]): DataFrame = {
  val drops = df.columns.diff(attrs)
  var newdf = df
  for(d<-drops){
    newdf = newdf.drop(d)   
  }
  newdf
}

def indexStringColumns(df:DataFrame, cols:Array[String]):DataFrame = {
  var newdf = df
  for(col <- cols) {
    val si = new
    StringIndexer().setInputCol(col).setOutputCol(col+"-numeric")
    val sm:StringIndexerModel = si.fit(newdf)
    println(col)
    sm.labels.foreach(println)
    println("--------------------")
    newdf = sm.transform(newdf).drop(col)
    newdf = newdf.withColumnRenamed(col+"-numeric", col)
  }
  newdf
}

/*
 *          Leyendo datos
 */

val dataRDD = getStringRDD(PATH,DATA)

val dataRDDNumeric = dataRDD.map(a=>Array(a(0).toDouble,a(1),a(2).toDouble,a(3),a(4).toDouble,a(5),a(6),a(7),a(8),a(9),a(10).toDouble,a(11).toDouble,a(12).toDouble,a(13),a(14)))

val dataDF = spark.createDataFrame(dataRDDNumeric.map(Row.fromSeq(_)),adultSchema)

/*
 *     Selección de atributos
 *        Podemos eliminar ciertos atributos
 */

val ATTR = Array("age","workclass","education","marital-status","relationship","race","sex","capital-gain","capital-loss", "label")
val dataSimpleDF = selectAttributes(dataDF,ATTR)
val CATATTR = ATTR.diff(CONTATTR)

/*
 *  Transformamos el DataFrame en uno con columna label y features, para lo cual debemos convertir los strings en doubles. 
 *  Utilizamos un StringIndexer.
 */

val numdataDF = indexStringColumns(dataSimpleDF,CATATTR)

val assembler = new VectorAssembler()
assembler.setOutputCol("features")
assembler.setInputCols(numdataDF.columns.diff(Array("label")))

val mldataDF = assembler.transform(numdataDF).select("features","label")

/*
 * Creamos un árbol de decisión con parámetros prefijados
 *  IMPORTANTE: El valor maxBins debe valer el menos como el mayor número de distintos valores categoricos
 */
val DT=new DecisionTreeClassifier()
val impureza = "entropy"
val maxBins = 16 

DT.setImpurity(impureza)
DT.setMaxBins(maxBins)

val meval = new MulticlassClassificationEvaluator()
meval.setMetricName("accuracy")

val crossval = new CrossValidator().setEstimator(DT).setEvaluator(meval)

val paramGrid = new ParamGridBuilder().addGrid(DT.maxDepth, Array(1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20)).build()
crossval.setEstimatorParamMaps(paramGrid)
crossval.setNumFolds(3)

val cvModel = crossval.fit(mldataDF)

val DTmodel = cvModel.bestModel
DTmodel.extractParamMap()