/* 
*       Lee y evalua datos de test mediante una pipeline de transformación y un modelo creado y almacenado
*/

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.CrossValidatorModel

import org.apache.spark.ml.feature.StringIndexerModel

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.NaiveBayesModel

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

/*
 *          Leyendo datos
 */

val testDF = getDF(PATH,TEST)

//En este caso la seleccion de atributos no es necesaria, ya que VectorAssembler utilizará solo aquellos atributos utilizados en Training

/*
 *                  TREE
 */


val cv = CrossValidatorModel.load(TREEPATH)
val pipe = cv.bestModel.asInstanceOf[PipelineModel]

println("ÁRBOL DE DECISIÓN:")
println("------------------------------------------")

//Podemos visualizar el árbol accediendo a la última etapa de la Pipeline
val dt:DecisionTreeClassificationModel = pipe.stages.last.asInstanceOf[DecisionTreeClassificationModel]
println(s"Altura del árbol: ${dt.depth}")

//Extraemos el StringIndexer de la clase, ya que no sabemos a qué etiqueta corresponde 0 y a cual 1
val si_clase = pipe.stages(pipe.stages.length-4).asInstanceOf[StringIndexerModel]

val clases = si_clase.labels
val predictionsAndLabelsDF = pipe.transform(testDF).select("label_numeric","prediction")
val predictionsAndLabelsRDD = predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

val meval = new MulticlassMetrics(predictionsAndLabelsRDD)
val beval = new BinaryClassificationMetrics(predictionsAndLabelsRDD)

val error = 1-meval.accuracy
val std = scala.math.sqrt((error*(1-error))/predictionsAndLabelsRDD.count)
val z = 1.96

println(s"Tasa de error: ${error*100}%")
println(s"STD: ${std}")
println(s"Intervalo de confianza al 95%: [${error-z*std},${error+z*std}]")
println("")
println(s"Area bajo la curva ROC: ${beval.areaUnderROC}")

println(s"Tasa de falsos positivos para ${clases(0)}: ${meval.falsePositiveRate(0.0)*100}%")
println(s"Tasa de falsos positivos para ${clases(1)}: ${meval.falsePositiveRate(1.0)*100}%")
println(s"Tasa de aciertos positivos para ${clases(0)}: ${meval.truePositiveRate(0.0)*100}%")
println(s"Tasa de aciertos positivos para ${clases(1)}: ${meval.truePositiveRate(1.0)*100}%")
println("")
println(s"Precisión para ${clases(0)}: ${meval.precision(0.0)}")
println(s"Recall para ${clases(0)}: ${meval.recall(0.0)}")
println(s"Precisión para ${clases(1)}: ${meval.precision(1.0)}")
println(s"Recall para ${clases(1)}: ${meval.recall(1.0)}")
println(s"Area bajo la curva PR: ${beval.areaUnderPR}")

 /*
 *                  NAIVE BAYES
 *      Los dos experimentos son modelos distintos, se deben leer de distinta manera
 */

//Primer experimento:
//val pipe2 = PipelineModel.load(NBPATH)

//Segundo experimento:
val cv2 = CrossValidatorModel.load(NBPATH)

println("NAIVE BAYES:")
println("------------------------------------------")

println("Modelo NB:")
//Primer experimento:
//println(pipe2.stages.last.extractParamMap)
//val indexer_clase = pipe2.stages(pipe2.stages.length-3).asInstanceOf[StringIndexerModel]

//Segundo experimento:
val pipe2 = cv2.bestModel.asInstanceOf[PipelineModel]
println(pipe2.stages.last.extractParamMap)
val indexer_clase = pipe2.stages(pipe2.stages.length-4).asInstanceOf[StringIndexerModel]

val clases = indexer_clase.labels
val predictionsAndLabelsDF2 = pipe2.transform(testDF).select("label_numeric","prediction")
val predictionsAndLabelsRDD2 = predictionsAndLabelsDF2.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

val meval = new MulticlassMetrics(predictionsAndLabelsRDD2)
val beval = new BinaryClassificationMetrics(predictionsAndLabelsRDD2)

val error = 1-meval.accuracy
val std = scala.math.sqrt((error*(1-error))/predictionsAndLabelsRDD2.count)
println(s"Tasa de error: ${error*100}%")
println(s"STD: ${std}")
println(s"Intervalo de confianza al 95%: [${error-z*std},${error+z*std}]")
println("")
println(s"Area bajo la curva ROC: ${beval.areaUnderROC}")

println(s"Tasa de falsos positivos para ${clases(0)}: ${meval.falsePositiveRate(0.0)*100}%")
println(s"Tasa de falsos positivos para ${clases(1)}: ${meval.falsePositiveRate(1.0)*100}%")
println(s"Tasa de aciertos positivos para ${clases(0)}: ${meval.truePositiveRate(0.0)*100}%")
println(s"Tasa de aciertos positivos para ${clases(1)}: ${meval.truePositiveRate(1.0)*100}%")
println("")
println(s"Precisión para ${clases(0)}: ${meval.precision(0.0)}")
println(s"Recall para ${clases(0)}: ${meval.recall(0.0)}")
println(s"Precisión para ${clases(1)}: ${meval.precision(1.0)}")
println(s"Recall para ${clases(1)}: ${meval.recall(1.0)}")
println(s"Area bajo la curva PR: ${beval.areaUnderPR}")
