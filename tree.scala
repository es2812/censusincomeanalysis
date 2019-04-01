/* 
 * Creación de modelos de clasificación para el dataset Census Income Dataset de UCI.
 *
 * Autor: Esther Cuervo Fernández
 *        25-03-2019
 *
 */

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler, StandardScaler}
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.classification.DecisionTreeClassifier
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

val SELATTR = Array("age","fnlwgt","capital-loss","workclass","education","marital-status","relationship","sex","label")

val dataSimpleDF = selectAttributes(dataDF,SELATTR)
val SELCATATTR = SELATTR.diff(CONTATTR) //guardamos los atributos categóricos en un array, los necesitaremos para la transformación de datos más adelante

/*
*   Creamos las etapas de la Pipeline de transformación de los datos y modelo:
*         1) StringIndexer para convertir aquellos atributos categóricos de tipo String en Double.
*               Este modelo guarda los atributos numéricos en columnas con el nombre original 
*               concatenado con "_numeric". Deberemos tener esto en cuenta para las subsiguientes etapas.
*         2) VectorAssembler para construir los DataFrames de features, label.
*               De nuevo tener en cuenta que el atributo label se llamará "label_numeric"
*         3) StandardScaler para estandarizar los vectores de features.
*               Este modelo convierte la columna "features" en "scaled_features" 
*         M) DecisionTreeClassifer el modelo a utilizar.
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

//las columnas que entrarán en el Assembler serán las correspondientes a atributos continuos seleccionados unión a las categoricas seleccionadas con _numeric añadido al final:
val INCOLS = SELCATATTR.map(x=>x+"_numeric").union(SELATTR.diff(SELCATATTR)).diff(Array("label_numeric"))

var assembler = new VectorAssembler().setOutputCol("features").setInputCols(INCOLS)

// 3)
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").setWithStd(true).setWithMean(false)

// M)
//IMPORTANTE: El valor maxBins debe valer el menos como el mayor número de distintos valores categoricos
// Recordar que las transformaciones resultan en un DF con scaled_features y label_numeric

val impureza = "entropy"
val maxBins = 16 
val DT=new DecisionTreeClassifier().setFeaturesCol("scaled_features").setLabelCol("label_numeric").setImpurity(impureza).setMaxBins(maxBins)

/*
*     Creamos el experimento de Validación Cruzada. Utiliza como estimador la pipeline, y como evaluador un MulticlassClassificationEvaluator 
*     con metrica "accuracy"
*/

val pipeline = new Pipeline().setStages((sims:::List(assembler,scaler,DT)).toArray)
// Es necesario decirle al evaluador que la etiqueta de clase es label_numeric
val meval = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("label_numeric")

val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(meval)
val paramGrid = new ParamGridBuilder().addGrid(DT.maxDepth, Array(1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20)).build()
crossval.setEstimatorParamMaps(paramGrid)
crossval.setNumFolds(5)

val cvModel = crossval.fit(dataSimpleDF)
cvModel.write.overwrite().save(TREECVPATH)
