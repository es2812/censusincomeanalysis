/* 
 * Creación de modelos de clasificación para el dataset Census Income Dataset de UCI. 
 *                          Naive Bayes Classifier
 *
 * Autor: Esther Cuervo Fernández
 *        31-03-2019
 *
 */

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, StringIndexerModel, OneHotEncoder, QuantileDiscretizer}
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/*
 *          Leyendo datos
 */

val dataDF = getDF(PATH,DATA)

/*
 *     Selección de atributos
 *        Podemos eliminar ciertos atributos. 
 *        NOTA: Debemos seleccionar manualmente "label"
 */

val SELATTR = ATTR

val dataSimpleDF = selectAttributes(dataDF,SELATTR)
val SELCATATTR = SELATTR.diff(CONTATTR) //guardamos los atributos categóricos en un array, los necesitaremos para la transformación de datos más adelante
val SELCONTATTR = SELATTR.diff(SELCATATTR)

/*
*   Creamos las etapas de la Pipeline de transformación de los datos y modelo:
*         1) StringIndexer para convertir aquellos atributos categóricos de tipo String en Double.
*               Este modelo guarda los atributos numéricos en columnas con el nombre original 
*               concatenado con "_numeric". Deberemos tener esto en cuenta para las subsiguientes etapas.
*         2) QuantileDiscretizer discretiza los atributos continuos. Se le debe dar un parámetro K para el número de conjuntos. Se aproximará con CV.
*               Guarda los atributos discretizados en columnas con nombre original concatenado con "_numeric"
*         3) VectorAssembler para construir los DataFrames de features, label.
*               De nuevo tener en cuenta que el atributo label se llamará "label_numeric"
*         M) NaiveBayes (Multinomial). 
*   Esta Pipeline debe ser introducida en CrossVal, para que se utilice para realizar la transformación de los conjuntos train y validation 
*   correctamente
*/

// 1)
var sims = List[StringIndexerModel]()

for(col<-SELCATATTR){
  val si = new StringIndexer().setInputCol(col).setOutputCol(col+"_numeric")
  val sm:StringIndexerModel = si.fit(dataSimpleDF)
  sims = sims:::List(sm)
}

// 2)
var OUTCOLS = SELCONTATTR.map(x=>x+"_numeric")
var qd = new QuantileDiscretizer().setInputCols(SELCONTATTR).setOutputCols(OUTCOLS)

// 3)
//las columnas que entrarán en el Assembler serán las correspondientes a atributos con _numeric añadido al final:
val INCOLS = SELATTR.map(x=>x+"_numeric").diff(Array("label_numeric"))
var assembler = new VectorAssembler().setOutputCol("features").setInputCols(INCOLS)

// M)
val NB=new NaiveBayes().setFeaturesCol("features").setLabelCol("label_numeric")

val pipeline = new Pipeline().setStages((sims:::List(qd,assembler,NB)).toArray)
val meval = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("label_numeric")

val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(meval)
val paramGrid = new ParamGridBuilder().addGrid(qd.numBuckets, Array(2,3,4,5,6,7,8,9, 10)).build()
crossval.setEstimatorParamMaps(paramGrid)
crossval.setNumFolds(5)

val cvModel = crossval.fit(dataSimpleDF)

println(s"Tasa de acierto media en cada experimento de Validación Cruzada:")
cvModel.avgMetrics.foreach(println)
cvModel.write.overwrite().save(NBCVPATH)
