1. Primer experimento: StringIndexer, todos los atributos. No nulls. DT entropy, 41 maxBin CrossVal con 5 folds de profundidad entre 1 y 20: Profundidad 8

Tasa de acierto: 84.03718459495352%

Tasa de falsos positivos para <=50k: 24.24483306836248%
Tasa de falsos positivos para >50k: 14.301658163265307%
Tasa de aciertos positivos para <=50k: 85.6983418367347%
Tasa de aciertos positivos para >50k: 75.75516693163752%

Area bajo la curva ROC: 0.807267543841861
Area bajo la curva PR: 0.4729406323773212

1. Segundo experimento: StringIndexer, 8 atributos ("age","education","education-num","marital-status","occupation","relationship","sex","hours-per-week"), no nulls. crossVal con 3 folds de profundidad entre 1 y 20 maxBins=16: Profundidad 9
 
Tasa de acierto: 28.751660026560426%

Tasa de falsos positivos para <=50k: 77.41381999688036%
Tasa de falsos positivos para >50k: 35.924932975871315%
Tasa de aciertos positivos para <=50k: 64.07506702412869%
Tasa de aciertos positivos para >50k: 22.586180003119637%

Area bajo la curva ROC: 0.4333062351362416
Area bajo la curva PR: 0.8092911447877731

3. Tercer experimento: StringIndexer, ("age","workclass","education","marital-status","relationship","race","sex","label"), no nulls. crossVal con 3 folds de profundidad entre 1 y 20 maxbins=16: Profundidad 9

Tasa de acierto: 82.19787516600266%

Tasa de falsos positivos para <=50k: 32.792975346166834%
Tasa de falsos positivos para >50k: 14.133399454500372%

Tasa de aciertos positivos para <=50k: 85.86660054549962%
Tasa de aciertos positivos para >50k: 67.20702465383317%

Area bajo la curva ROC: 0.765368125996664
Area bajo la curva PR: 0.48188903885903117

4. Experimento 4: lo mismo que arriba pero con `fnlwgt` añadido. Profundidad 8

Tasa de acierto: 81.83266932270917%

Tasa de falsos positivos para <=50k: 33.53825136612022%
Tasa de falsos positivos para >50k: 14.457632706890866%

Tasa de aciertos positivos para <=50k: 85.54236729310914%
Tasa de aciertos positivos para >50k: 66.46174863387978%

Area bajo la curva ROC: 0.7600205796349446
Area bajo la curva PR: 0.4703523308920572

5. Experimento 5: lo mismo que 3 pero añadiendo `capital-gain` y `capital-loss`. Profundidad 12:

Tasa de acierto: 83.38645418326693%

Tasa de falsos positivos para <=50k: 29.818059299191376%
Tasa de falsos positivos para >50k: 13.372477671187562%

Tasa de aciertos positivos para <=50k: 86.62752232881243%
Tasa de aciertos positivos para >50k: 70.18194070080862%

Area bajo la curva ROC: 0.7840473151481053
Area bajo la curva PR: 0.508421635632744

Best so far: 5. TODO: eliminar los outliers de `capital-gain`