/* 
*       Lee y evalua datos de test mediante una pipeline de transformación y un modelo creado y almacenado
*/

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.CrossValidatorModel

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.NaiveBayesModel

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

/*
 *          Leyendo datos
 */

val testDF = getDF(PATH,TEST)

//En este caso la seleccion de atributos no es necesaria, ya que VectorAssembler utilizará solo aquellos atributos utilizados en Training

// /*
//  *                  TREE
//  */

// val pipe = PipelineModel.load(TREEPATH)

// println("ÁRBOL DE DECISIÓN:")
// println("------------------------------------------")

// //Podemos visualizar el árbol accediendo a la última etapa de la Pipeline
// val dt:DecisionTreeClassificationModel = pipe.stages.last.asInstanceOf[DecisionTreeClassificationModel]
// println(s"Altura del árbol: ${dt.depth}")
// //println(s"Árbol:")
// //print(dt.toDebugString)

// val predictionsAndLabelsDF = pipe.transform(testDF).select("label_numeric","prediction")
// val predictionsAndLabelsRDD = predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

// val meval = new MulticlassMetrics(predictionsAndLabelsRDD)
// val beval = new BinaryClassificationMetrics(predictionsAndLabelsRDD)

// val error = 1-meval.accuracy
// val std = scala.math.sqrt((error*(1-error))/predictionsAndLabelsRDD.count)
// println(s"Tasa de error: ${error*100}%")
// println(s"STD: ${std}")
// val z = 1.96
// println(s"Intervalo de confianza al 95%: [${error-z*std},${error+z*std}]")
// println("")
// println(s"Area bajo la curva ROC: ${beval.areaUnderROC}")
// println(s"Tasa de falsos positivos para <=50k: ${meval.falsePositiveRate(0.0)*100}%")
// println(s"Tasa de falsos positivos para >50k: ${meval.falsePositiveRate(1.0)*100}%")
// println(s"Tasa de aciertos positivos para <=50k: ${meval.truePositiveRate(0.0)*100}%")
// println(s"Tasa de aciertos positivos para >50k: ${meval.truePositiveRate(1.0)*100}%")
// println("")
// println(s"Precisión para <=50k: ${meval.precision(0.0)}")
// println(s"Recall para <=50k: ${meval.recall(0.0)}")
// println(s"Precisión para >50k: ${meval.precision(1.0)}")
// println(s"Recall para >50k: ${meval.recall(1.0)}")
// println(s"Area bajo la curva PR: ${beval.areaUnderPR}")

 /*
 *                  NAIVE BAYES
 */

//Primer experimento:
val pipe2 = PipelineModel.load(NBPATH)

//Segundo experimento:
// val pipe2 = CrossValidatorModel.load(NBPATH)

println("NAIVE BAYES:")
println("------------------------------------------")

println("Modelo NB:")
println(pipe2.stages.last.extractParamMap)

//Segundo experimento:
// println("QuantileDiscretizer:")
// println(pipe2.stages(pipe2.stages.length-3).extractParamMap)

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
println(s"Tasa de falsos positivos para <=50k: ${meval.falsePositiveRate(0.0)*100}%")
println(s"Tasa de falsos positivos para >50k: ${meval.falsePositiveRate(1.0)*100}%")
println(s"Tasa de aciertos positivos para <=50k: ${meval.truePositiveRate(0.0)*100}%")
println(s"Tasa de aciertos positivos para >50k: ${meval.truePositiveRate(1.0)*100}%")
println("")
println(s"Precisión para <=50k: ${meval.precision(0.0)}")
println(s"Recall para <=50k: ${meval.recall(0.0)}")
println(s"Precisión para >50k: ${meval.precision(1.0)}")
println(s"Recall para >50k: ${meval.recall(1.0)}")
println(s"Area bajo la curva PR: ${beval.areaUnderPR}")
