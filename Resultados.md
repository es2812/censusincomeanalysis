# Decision Tree

## Primer experimento

- StringIndexer
- Sin selección de atributos
- No nulls
- Entropy, 41 maxBin 
- CrossVal con 5 folds de profundidad entre 1 y 20

+ Depth: 8
+ Tasa de acierto: 84.5484727755644%
+ Area bajo la curva ROC: 0.820364659706015
+ Tasa de falsos positivos para <=50k: 21.667354519191086%
+ Tasa de falsos positivos para >50k: 14.25971353960592%
+ Tasa de aciertos positivos para <=50k: 85.74028646039407%
+ Tasa de aciertos positivos para >50k: 78.33264548080892%
+ Precisión para <=50k: 0.9537852112676056
+ Recall para <=50k: 0.8574028646039408
+ Precisión para >50k: 0.512972972972973
+ Recall para >50k: 0.7833264548080892
+ Area bajo la curva PR: 0.47482941553659097

## Segundo experimento

- StringIndexer
- 8 atributos ("age","education","education-num","marital-status","occupation","relationship","sex","hours-per-week")
- No nulls
- Entropy, 16 maxBin 
- CrossVal con 5 folds de profundidad entre 1 y 20

+ Depth: 9
+ Tasa de acierto: 82.70252324037185%
+ Tasa de falsos positivos para <=50k: 32.89597000937207%
+ Tasa de falsos positivos para >50k: 13.087106838687918%
+ Tasa de aciertos positivos para <=50k: 86.91289316131208%
+ Tasa de aciertos positivos para >50k: 67.10402999062794%
+ Area bajo la curva ROC: 0.7700846157597001
+ Precisión para <=50k: 0.9073063380281691
+ Recall para <=50k: 0.8691289316131209
+ Precisión para >50k: 0.5805405405405405
+ Recall para >50k: 0.6710402999062793
+ Area bajo la curva PR: 0.520013478848859


## Tercer experimento

- StringIndexer
- 9 atributos ("age","fnlwgt","capital-gain","capital-loss","workclass","education","marital-status","relationship","sex")
- No nulls
- Entropy, 16 maxBin 
- CrossVal con 5 folds de profundidad entre 1 y 20

+ Depth: 9
+ Tasa de acierto: 83.17397078353254%
+ Area bajo la curva ROC: 0.7818548743748641
+ Tasa de falsos positivos para <=50k: 29.938059187887127%
+ Tasa de falsos positivos para >50k: 13.690965937140037%
+ Tasa de aciertos positivos para <=50k: 86.30903406285996%
+ Tasa de aciertos positivos para >50k: 70.06194081211287%
+ Precisión para <=50k: 0.9234154929577465
+ Recall para <=50k: 0.8630903406285997
+ Precisión para >50k: 0.5502702702702703
+ Recall para >50k: 0.7006194081211287
+ Area bajo la curva PR: 0.4967846128182347

Como se puede ver la clase mayoritaria (<=50k) tiene un correcto resultado, para >50k, sin embargo, la precisión es muy baja, aunque recall es bueno. Esto quiere decir que consigue clasificar un 70% de las instancias de >50k, sin embargo solo un 55% de las que clasifica como >50k lo son.


## Cuarto experimento

- StringIndexer
- 8 atributos ("age","fnlwgt","capital-loss","workclass","education","marital-status","relationship","sex")
- No nulls
- Entropy, 16 maxBin 
- CrossVal con 5 folds de profundidad entre 1 y 20

+ Depth: 9
+ Tasa de acierto: 82.04515272244356%
+ Area bajo la curva ROC: 0.7648615160189981
+ Tasa de falsos positivos para <=50k: 32.42766407904023%
+ Tasa de falsos positivos para >50k: 14.60003271716015%
+ Tasa de aciertos positivos para <=50k: 85.39996728283985%
+ Tasa de aciertos positivos para >50k: 67.57233592095977%
+ Precisión para <=50k: 0.9191021126760563
+ Recall para <=50k: 0.8539996728283985
+ Precisión para >50k: 0.5175675675675676
+ Recall para >50k: 0.6757233592095977
+ Area bajo la curva PR: 0.4641613196517433

Peor

## Quinto experimento

- StringIndexer
- Todos los atributos PCA
- No nulls
- Entropy, 5 maxBin
- CrossVal con 2 folds de profundidad entre 7 y 14 para maxDepth de Tree y 5 hasta 9 para K de PCA. Utiliza AreaUnderROC para elegir el modelo

+ Depth: 14
+ K: 9
+ Varianza explicada total: 77%
+ Tasa de acierto: 80.90305444887119%
+ Area bajo la curva ROC: 0.7440848369614439
+ Tasa de falsos positivos para <=50k: 36.52714192282537%
+ Tasa de falsos positivos para >50k: 14.655890684885852%
+ Tasa de aciertos positivos para <=50k: 85.34410931511415%
+ Tasa de aciertos positivos para >50k: 63.47285807717462%
+ Precisión para <=50k: 0.9016725352112676
+ Recall para <=50k: 0.8534410931511415
+ Precisión para >50k: 0.5245945945945946
+ Recall para >50k: 0.6347285807717462
+ Area bajo la curva PR: 0.46586988191096934


## Quinto experimento (II)

- StringIndexer
- Todos los atributos PCA
- No nulls
- Entropy, 5 maxBin
- CrossVal con 2 folds de profundidad entre 7 y 14 para maxDepth de Tree y 5 hasta 13 para K de PCA. Utiliza AreaUnderROC para elegir el modelo

+ Depth: 14
+ K: 13
+ Varianza explicada total: 97%
+ Tasa de acierto: 80.4316069057105%
+ Area bajo la curva ROC: 0.7357804593238797
+ Tasa de falsos positivos para <=50k: 38.824577025823686%
+ Tasa de falsos positivos para >50k: 14.019331109400394%
+ Tasa de aciertos positivos para <=50k: 85.98066889059962%
+ Tasa de aciertos positivos para >50k: 61.17542297417632%
+ Precisión para <=50k: 0.8848591549295775
+ Recall para <=50k: 0.8598066889059961
+ Precisión para >50k: 0.557027027027027
+ Recall para >50k: 0.6117542297417632
+ Area bajo la curva PR: 0.4923216282663623

PCA no parece útil en este caso, mejor utilizar atributos.

## Sexto experimento

- StringIndexer
- 8 atributos ("age","fnlwgt","capital-loss","workclass","education","marital-status","relationship","sex")
- No nulls
- Entropy, 5 maxBin 
- CrossVal con 5 folds de profundidad entre 5 y 14, utilizando areaUnderROC para elección de modelo

+ Depth: 11
+ Tasa de acierto: 81.64674634794157%
+ Area bajo la curva ROC: 0.755676082447736
+ Tasa de falsos positivos para <=50k: 34.645669291338585%
+ Tasa de falsos positivos para >50k: 14.219114219114218%
+ Tasa de aciertos positivos para <=50k: 85.78088578088578%
+ Tasa de aciertos positivos para >50k: 65.35433070866141%
+ Precisión para <=50k: 0.9070422535211268
+ Recall para <=50k: 0.8578088578088578
+ Precisión para >50k: 0.5383783783783784
+ Recall para >50k: 0.6535433070866141
+ Area bajo la curva PR: 0.48017574308003147


Ideas: considerar capital-gain o loss, ya que en muchos casos es 0. education-num y education son redundantes. fnlwgt es un parámetro con demasiados valores distintos, native-country y race tienen muchos valores distintos pero las proporciones son, en todos los casos muy desbalanceadas (mayoría de americanos y blancos)

# NAIVE BAYES

## Primer Experimento

- StringIndexer (Multinomial)
- Utiliza todos los atributos categóricos
- No nulls

+ Tasa de acierto: 71.75298804780877%
+ Area bajo la curva ROC: 0.6082208240855652
+ Tasa de falsos positivos para <=50k: 58.48132271892222%
+ Tasa de falsos positivos para >50k: 19.874512463964727%
+ Tasa de aciertos positivos para <=50k: 80.12548753603527%
+ Tasa de aciertos positivos para >50k: 41.51867728107777%
+ Precisión para <=50k: 0.8318661971830986
+ Recall para <=50k: 0.8012548753603527
+ Precisión para >50k: 0.36648648648648646
+ Recall para >50k: 0.4151867728107777
+ Area bajo la curva PR: 0.3227364286530195

## Segundo Experimento

- StringIndexer (Multinomial)
- Todos los atributos. Utilizamos QuantileDiscretizer con numBuckets estimado por CrossVal para el número de valores discretos (2 hasta 10 5 folds) Array[Array[Double]] = Array(Array(-Infinity, 28.0, 37.0, 47.0, Infinity), Array(-Infinity, 117606.0, 178417.0, 236992.0, Infinity), Array(-Infinity, 9.0, 10.0, 13.0, Infinity), Array(-Infinity, 0.0, Infinity), Array(-Infinity, 0.0, Infinity), Array(-Infinity, 40.0, 45.0, Infinity))
- No nulls

+ Tasa de acierto: 77.05843293492696%
+ Area bajo la curva ROC: 0.6998682026382422
+ Tasa de falsos positivos para <=50k: 47.240369452579415%
+ Tasa de falsos positivos para >50k: 12.78599001977215%
+ Tasa de aciertos positivos para <=50k: 87.21400998022784%
+ Tasa de aciertos positivos para >50k: 52.759630547420585%
+ Precisión para <=50k: 0.8154049295774648
+ Recall para <=50k: 0.8721400998022785
+ Precisión para >50k: 0.632972972972973
+ Recall para >50k: 0.5275963054742059
+ Area bajo la curva PR: 0.5530851014334921
