/* 
*       Lee y evalua un modelo creado con datos almacenados en un DataFrame mltestDF ya en la sesión
*/

import org.apache.spark.rdd.RDD

import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType,IntegerType}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.{StringIndexer,StringIndexerModel}
import org.apache.spark.ml.feature.VectorAssembler


import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

/*
 *          Algunas definiciones
 */
val PATH="./datos/"
val TEST="adult.test"

val ATTR= Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss", "hours-week", 
  "native-country", "label")
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
val testRDD = getStringRDD(PATH,TEST)
val testRDDNumeric = testRDD.map(a=>Array(a(0).toDouble,a(1),a(2).toDouble,a(3),a(4).toDouble,a(5),a(6),a(7),a(8),a(9),a(10).toDouble,a(11).toDouble,a(12).toDouble,a(13),a(14)))
val testDF = spark.createDataFrame(testRDDNumeric.map(Row.fromSeq(_)),adultSchema)

/*
 *     Selección de atributos
 *        Podemos eliminar ciertos atributos
 */

val ATTR = Array("age","workclass","education","marital-status","relationship","race","sex","capital-gain","capital-loss", "label")
val testSimpleDF = selectAttributes(testDF,ATTR)
val CATATTR = ATTR.diff(CONTATTR)

/*
 *  Transformamos el DataFrame en uno con columna label y features, para lo cual debemos convertir los strings en doubles. 
 *  Utilizamos un StringIndexer.
 */

val numtestDF = indexStringColumns(testSimpleDF,CATATTR)

val assembler = new VectorAssembler()
assembler.setOutputCol("features")
assembler.setInputCols(numtestDF.columns.diff(Array("label")))

val mltestDF = assembler.transform(numtestDF).select("features","label")


/*
*   Lectura del modelo
*/

val PATH = "models/experimento5/bestModel"
val treeModel = DecisionTreeClassificationModel.load(PATH)


treeModel.toDebugString

val predictionsAndLabelsDF = treeModel.transform(mltestDF).select("label","prediction")
val predictionsAndLabelsRDD=predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

val meval = new MulticlassMetrics(predictionsAndLabelsRDD)
val beval = new BinaryClassificationMetrics(predictionsAndLabelsRDD)

println(s"Tasa de acierto: ${meval.accuracy*100}%")
println(s"Tasa de falsos positivos para <=50k: ${meval.falsePositiveRate(0.0)*100}%")
println(s"Tasa de falsos positivos para >50k: ${meval.falsePositiveRate(1.0)*100}%")
println(s"Tasa de aciertos positivos para <=50k: ${meval.truePositiveRate(0.0)*100}%")
println(s"Tasa de aciertos positivos para >50k: ${meval.truePositiveRate(1.0)*100}%")
println(s"Area bajo la curva ROC: ${beval.areaUnderROC}")
println(s"Area bajo la curva PR: ${beval.areaUnderPR}")
