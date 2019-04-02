/* 
*       Incluye algunas definiciones necesarias en ficheros nb, tree y readandeval
*/

import org.apache.spark.sql.types.{StructType, StructField, StringType, DoubleType}
import org.apache.spark.sql.{DataFrame, Row}


/*
*              CONSTANTES
*/

val PATH="./datos/"
val DATA="adult.data"
val TEST="adult.test"

val ATTR= Array("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss", "hours-week", "native-country", "label") //atributos en dataset 

val CONTATTR = Array("age","fnlwgt","education-num","capital-gain","capital-loss","hours-week")
val CATATTR = ATTR.diff(CONTATTR)

val TREEPATH = "models/TREE/experimento1"
val NBPATH = "models/NB/experimento1"

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
*               FUNCIONES
*/

// lee y transforma el dataset en un dataframe utilizando el schema definido en este mismo fichero
def getDF(path:String, file:String):DataFrame = {
    // los valores faltantes se representan con "?", lo podemos cambiar por null para poder leer los datos a un DF
    // sin embargo para la primera aproximación los eliminamos
    val parsed = sc.textFile(path+file).filter(_.nonEmpty).map(line=>line.split(", ")).filter(x=>{!(x.contains("?"))})

    // algunos valores son numéricos, los convertimos antes de aplicar el schema
    val numeric = parsed.map(a=>Array(a(0).toDouble,a(1),a(2).toDouble,a(3),a(4).toDouble,a(5),a(6),a(7),a(8),a(9),a(10).toDouble,a(11).toDouble,a(12).toDouble,a(13),a(14)))

    spark.createDataFrame(numeric.map(Row.fromSeq(_)),adultSchema)
}

// elimina de df aquellas columnas no en attrs
def selectAttributes(df:DataFrame, attrs:Array[String]): DataFrame = {
    val drops = df.columns.diff(attrs)
    var newdf = df
    for(d<-drops){
        newdf = newdf.drop(d)   
    }
    newdf
}
