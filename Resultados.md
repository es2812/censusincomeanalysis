# Experimentos

## Primer experimento

- StringIndexer
- Sin selección de atributos
- No nulls
- Entropy, 41 maxBin 
- CrossVal con 5 folds de profundidad entre 1 y 20

+ Depth: 8
+ Tasa de acierto: 84.5484727755644%
+ Tasa de falsos positivos para <=50k: 21.667354519191086%
+ Tasa de falsos positivos para >50k: 14.25971353960592%
+ Tasa de aciertos positivos para <=50k: 85.74028646039407%
+ Tasa de aciertos positivos para >50k: 78.33264548080892%
+ Area bajo la curva ROC: 0.820364659706015
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
+ Tasa de falsos positivos para <=50k: 29.938059187887127%
+ Tasa de falsos positivos para >50k: 13.690965937140037%
+ Tasa de aciertos positivos para <=50k: 86.30903406285996%
+ Tasa de aciertos positivos para >50k: 70.06194081211287%
+ Precisión para <=50k: 0.9234154929577465
+ Recall para <=50k: 0.8630903406285997
+ Precisión para >50k: 0.5502702702702703
+ Recall para >50k: 0.7006194081211287
+ Area bajo la curva ROC: 0.7818548743748641
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

TODO: hacer PCA tras **escalar**
Ideas: considerar capital-gain o loss, ya que en muchos casos es 0. education-num y education son redundantes. fnlwgt es un parámetro con demasiados valores distintos, native-country y race tienen muchos valores distintos pero las proporciones son, en todos los casos muy desbalanceadas (mayoría de americanos y blancos)