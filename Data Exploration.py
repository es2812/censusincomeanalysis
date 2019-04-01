#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'TEAD\MiniProyecto1\censusincomeanalysis'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# - 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
# - 45222 if instances with unknown values are removed (train=30162, test=15060)
# - Duplicate or conflicting instances : 6
# - Class probabilities for adult.all file
#     - Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
#     - Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
# 
# Prediction task is to determine whether a person makes over 50K a year.
# 
# Error Accuracy reported as follows, after removal of unknowns from train/test sets):
# 
# |     | Algorithm             | Error                                        |
# |-----|-----------------------|----------------------------------------------|
# | 1   | C4.5                  | 15.54                                        |
# | 2   | C4.5-auto             | 14.46                                        |
# | 3   | C4.5 rules            | 14.94                                        |
# | 4   | Voted ID3 (0.6)       | 15.64                                        |
# | 5   | Voted ID3 (0.8)       | 16.47                                        |
# | 6   | T2                    | 16.84                                        |
# | 7   | 1R                    | 19.54                                        |
# | 8   | NBTree                | 14.10                                        |
# | 9   | CN2                   | 16.00                                        |
# | 10  | HOODG                 | 14.82                                        |
# | 11  | FSS Naive Bayes       | 14.05                                        |
# | 12  | IDTM (Decision table) | 14.46                                        |
# | 13  | Naive-Bayes           | 16.12                                        |
# | 14  | Nearest-neighbor (1)  | 21.42                                        |
# | 15  | Nearest-neighbor (3)  | 20.35                                        |
# | 16  | OC1                   | 15.04                                        |
# | 17  | Pebls                 | Crashed. Unknown why (bounds WERE increased) |
# 
# Description of fnlwgt (final weight)
# 
# The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US.  These are prepared monthly for us by Population Division here at the Census Bureau.  We use 3 sets of
# controls. These are:
# 1.  A single cell estimate of the population 16+ for each state.
# 2.  Controls for Hispanic Origin by age and sex.
# 3.  Controls by Race, age and sex.
# 
# We use all three sets of controls in our weighting program and "rake" through them 6 times so that by the end we come back to all the controls we used.
# 
# The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population.
# 
# People with similar demographic characteristics should have similar weights.  There is one important caveat to remember about this statement.  That is that since the CPS sample is actually a collection of 51 state samples, each with its own probability of selection, the statement only applies within state.
# 
# - **age**: continuous.
# - **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - **fnlwgt**: continuous.
# - **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - **education-num**: continuous. number of years studied.
# - **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - **sex**: Female, Male.
# - **capital-gain**: continuous.
# - **capital-loss**: continuous.
# - **hours-per-week**: continuous.
# - **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# - **Class**: >50K, <=50K.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('pylab', 'inline')

#%% [markdown]
# # Lectura y resumen de datos

#%%
headers = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
            "relationship","race","sex","capital-gain","capital-loss","hours-per-week",
            "native-country","CLASS"]
continuos = {"age":True,"workclass":False,"fnlwgt":True,"education":False,"education-num":True,"marital-status":False,
             "occupation":False,"relationship":False,"race":False,"sex":False,"capital-gain":True,"capital-loss":True,
             "hours-per-week":True,"native-country":False,"CLASS":False}
data = pd.read_csv("./datos/adult.data",skipinitialspace=True,header=None,names=headers)


#%% [markdown]
# Los valores desconocidos se representan por un `?`, lo cambiamos por np.nan (valor desconocido en pandas)

#%%
for col in data.columns:
    data[col] = data[col].map(lambda x: x if x!='?' else np.nan)

#%% [markdown]
# Describimos cada atributo mediante su media, desviación típica, valor mínimo y máximo en el caso de atributos numéricos y valores únicos y moda en el caso de los atributos continuos

#%%
for col in data.drop('CLASS',axis=1).columns:
    print("Atributo %s:" % col)
    if(continuos[col]):
        print("\tMedia: %f, Desv. Tip: %f, Min: %f, Max: %f" % (data[col].mean(),data[col].std(),data[col].min(),data[col].max()))
    else:
        uniq = data[col].dropna().unique()
        for u in uniq:
            print("\t%s: %d" % (u,data[col].where(lambda x: x==u).count()))
        print("\tMISSING VALUES: %d" % (data[col].isnull().sum()))

#%% [markdown]
# # Representación de atributos

#%%
data_class1 = data.where(data.CLASS=='<=50K')
data_class2 = data.where(data.CLASS=='>50K')

#%% [markdown]
# ## age

#%%
X, Y = np.unique(data.age,return_counts=True)


#%%
pylab.rcParams['figure.figsize'] = (15, 5)
fig,(ax1, ax2) = plt.subplots(1,2)

#histograma
ax1.set_title("Frecuencia del atributo age")
ax1.set_xlabel("age")
ax1.set_ylabel("Frecuencia")
ax1.bar(X,Y)

#gráfico de caja y bigote por clases
ax2.set_title("Boxplot")
class1 = data_class1.dropna().age
class2 = data_class2.dropna().age
ax2.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
ax2.set_xlabel("age")
ax2.set_ylabel("Clase")
plt.show()

#%% [markdown]
# En ambos gráficos se ve como los valores por encima de 70 son mucho menos frecuentes, y el boxplot muestra, además, la distribución de los valores por clase. Se puede ver que de media los ciudadanos de rentas bajas son más jóvenes que los de rentas altas.
#%% [markdown]
# ## workclass

#%%
X_1, Y_1 = np.unique(data_class1.workclass.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.workclass.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo workclass por clases")
ax1.set_xlabel("Workclass")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de workclass en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de workclass en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Como se puede observar, `Private` es el valor más frecuente, y en ambas clases la distribución es similar
#%% [markdown]
# ## fnlwgt

#%%
pylab.rcParams['figure.figsize'] = (15, 5)

#gráfico de caja y bigote por clases
plt.title("Boxplot")
class1 = data_class1.dropna().fnlwgt
class2 = data_class2.dropna().fnlwgt
plt.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
plt.xlabel("weight")
plt.ylabel("Clase")
plt.show()

#%% [markdown]
# Se requiere una investigación de lo que significa este atributo antes de poder explicarlo
#%% [markdown]
# ## education

#%%
X_1, Y_1 = np.unique(data_class1.education.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.education.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo education por clases")
ax1.set_xlabel("Education")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de education en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de education en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Como se puede observar, `High School Graduate` es el valor más frecuente, seguido de `Some-college` y `Bachelors`. En este caso, el porcentaje de `Bachelors` en sueldos altos es bastante mayor, y en sueldos menores son más significativos `7th-8th` y `11th`
#%% [markdown]
# También es destacable el bajo porcentaje de `Masters` y `Doctorates` en sueldos bajos.
#%% [markdown]
# Todo esto tiene sentido, ya que los graduados, masters y doctores se espera que tengan un mayor sueldo, mientras que habrá muy poca gente con educación superior y sueldos menores
#%% [markdown]
# ## education-num

#%%
X, Y = np.unique(data["education-num"],return_counts=True)


#%%
pylab.rcParams['figure.figsize'] = (15, 5)
fig,(ax1, ax2) = plt.subplots(1,2)

#histograma
ax1.set_title("Frecuencia del atributo education-num")
ax1.set_xlabel("education-num")
ax1.set_ylabel("Frecuencia")
ax1.bar(X,Y)

#gráfico de caja y bigote por clases
ax2.set_title("Boxplot")
class1 = data_class1.dropna()["education-num"]
class2 = data_class2.dropna()["education-num"]
ax2.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
ax2.set_xlabel("education-num")
ax2.set_ylabel("Clase")
plt.show()

#%% [markdown]
# En este atributo se ve una clara predominación por entre 9 y 10 años estudiados, con 13 siendo un número también significativo, sin embargo, por clases, se puede observar que en sueldos bajos es donde se aglomeran los primeros dos valores (9 y 10), con muy poca incidencia del resto de valores, y el sueldo superior está más repartido entre los valores, con una prevalencia de un número de años de estudio mayor
#%% [markdown]
# ## marital-status

#%%
X_1, Y_1 = np.unique(data_class1["marital-status"].dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2["marital-status"].dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo marital-status por clases")
ax1.set_xlabel("Marital-status")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de marital-status en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de marital-status en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Este atributo tiene unas distribuciones muy distintas entre clases, con personas con sueldos bajos siendo mayoritariamente nunca casados, seguidos de casados con pareja no militar. Sin embargo en sueldos altos la gran mayoría de peresonas están casadas con pareja no militar.
#%% [markdown]
# También es significativo el alto porcentaje de divorciados y viudos en sueldos bajos, comparado con sueldos altos.
#%% [markdown]
# ## occupation

#%%
X_1, Y_1 = np.unique(data_class1.occupation.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.occupation.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo occupation por clases")
ax1.set_xlabel("Occupation")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+100, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de occupation en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de occupation en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Como podemos ver este atributo tiene un reparto relativamente igualitario en sueldos bajos, mientras que en sueldos altos la mayoría son `Exec-managerial`, `Prof-specialty` y `Sales`, aunque la misma proporción de este último se encuentra en sueldos bajos. 
#%% [markdown]
# En sueldos bajos se pueden ver algunas profesiones más comunes, como son `Adm-clerical`, `Craft-repair`, `Other-service` y `Sales`
#%% [markdown]
# ## relationship

#%%
X_1, Y_1 = np.unique(data_class1.relationship.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.relationship.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo relationship por clases")
ax1.set_xlabel("Relationship")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de relationship en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de relationship en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Como se puede observar, en sueldos altos es muy común ser `Husband`, mientras que en sueldos bajos está repartido entre `Husband` y `Not-in-family`, con cierto porcentaje de `Own-child` y `Unmarried`. Esto claramente denota lo que ya mostraba `marital-status`, los sueldos altos se aglomeran entre personas casadas (hombres, como ya desvela este atributo), y los sueldos bajos se reparten más igualitariamente, existiendo muchos solteros y gente que no es parte de una familia.
#%% [markdown]
# ## race

#%%
X_1, Y_1 = np.unique(data_class1.race.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.race.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo race por clases")
ax1.set_xlabel("Race")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de race en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de race en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Este atributo no es excesivamente significativo, ya que la mayoría de la población censada es blanca. Se puede observar una cierta mayoría de gente negra en sueldos bajos, pero no de manera demasiado significativa
#%% [markdown]
# ## sex

#%%
X_1, Y_1 = np.unique(data_class1.sex.dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2.sex.dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo sex por clases")
ax1.set_xlabel("Sex")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de sex en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de sex en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Como se puede observar, el sexo marca una diferencia significativa, con las mujeres formando un alto porcentaje de personas que reciben sueldo bajo, mientras que un gran porcentaje de personas con sueldo alto son hombres.

#%%
plt.title("Porcentaje de clase por valor del atributo sex")

female_1 = data_class1.where(data_class1.sex=="Female").dropna().age.count()
male_1 = data_class1.where(data_class1.sex=="Male").dropna().age.count()

female_2 = data_class2.where(data_class2.sex=="Female").dropna().age.count()
male_2 = data_class2.where(data_class2.sex=="Male").dropna().age.count()

y = [female_1/(female_1+male_1),female_2/(female_2+male_2)]
y2 = [male_1/(male_1+female_1),male_2/(female_2+male_2)]

plt.xlabel("Sex")
plt.ylabel("%")
plt.bar(X_1,y,label='<=50K')
plt.bar(X_2,y2,bottom=y,label='>50K')
plt.legend()

#%% [markdown]
# Como se puede ver, el 40% de las mujeres son de sueldo bajo, mientras que ese mismo número es de alrededor del 15% en hombres
#%% [markdown]
# ## capital-gain
#%% [markdown]
# **Este atributo tiene 153 elementos que valen 999999, lo cual parece ser un error. Eliminamos esos puntos**

#%%
data_class2 = data_class2.where(data_class2["capital-gain"]!=99999).dropna()


#%%
pylab.rcParams['figure.figsize'] = (15, 5)

#gráfico de caja y bigote por clases
plt.title("Boxplot")
class1 = data_class1.dropna()["capital-gain"]
class2 = data_class2.dropna()["capital-gain"]
plt.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
plt.xlabel("capital-gain")
plt.ylabel("Clase")
plt.show()

#%% [markdown]
# Se requiere una investigación de lo que significa este atributo antes de poder explicarlo
#%% [markdown]
# ## capital-loss

#%%
pylab.rcParams['figure.figsize'] = (15, 5)

#gráfico de caja y bigote por clases
plt.title("Boxplot")
class1 = data_class1.dropna()["capital-loss"]
class2 = data_class2.dropna()["capital-loss"]
plt.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
plt.xlabel("capital-loss")
plt.ylabel("Clase")
plt.show()

#%% [markdown]
# Se requiere una investigación de lo que significa este atributo antes de poder explicarlo
#%% [markdown]
# ## hours-per-week

#%%
X, Y = np.unique(data["hours-per-week"],return_counts=True)


#%%
pylab.rcParams['figure.figsize'] = (15, 5)
fig,(ax1, ax2) = plt.subplots(1,2)

#histograma
ax1.set_title("Frecuencia del atributo hours-per-week")
ax1.set_xlabel("horas")
ax1.set_ylabel("Frecuencia")
ax1.bar(X,Y)

#gráfico de caja y bigote por clases
ax2.set_title("Boxplot")
class1 = data_class1.dropna()['hours-per-week']
class2 = data_class2.dropna()['hours-per-week']
ax2.boxplot([class1,class2],vert=False,labels=["<=50K",">50K"])
ax2.set_xlabel("horas")
ax2.set_ylabel("Clase")
plt.show()

#%% [markdown]
# Se puede ver en ambos gráficos que la inmensa mayoría se sitúa en las 40 horas semanales, sobre todo en el caso de sueldos bajos. En sueldos altos este valor está más repartido, y es interesante observar que las medianas se situan, en ambos casos en 40 horas, pero en el caso de sueldos altos se trata también del primer quartil, mientras que en sueldos bajos coincide con el tercer cuartil. 
# 
# Esto indica que en sueldos altos los valores se distribuyen en y por encima de las 40 horas, mientras que en sueldos bajos en y por debajo
#%% [markdown]
# ## native-country

#%%
X_1, Y_1 = np.unique(data_class1["native-country"].dropna(),return_counts=True)
X_2, Y_2 = np.unique(data_class2["native-country"].dropna(),return_counts=True)


#%%
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 2 con 0s
for i,x1 in enumerate(X_1):
    if(not(np.isin(x1,X_2))):
        X_2 = np.insert(X_2,i,x1)
        Y_2 = np.insert(Y_2,i,0)
        
#esta función rellena las frecuencias de valores de los atributos que no existen para la clase 1 con 0s
for i,x2 in enumerate(X_2):
    if(not(np.isin(x2,X_1))):
        X_1 = np.insert(X_1,i,x2)
        Y_1 = np.insert(Y_1,i,0)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo native-country por clases")
ax1.set_xlabel("Country")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de native-country en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de native-country en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# Sin un primer filtrado, podemos observar que esta representación es inútil, ya que estados unidos es abrumadoramente común. Eliminamos ese país y trazamos otros gráficos:

#%%
Y_1 = Y_1[X_1 != "United-States"]
X_1 = X_1[X_1 != "United-States"]

Y_2 = Y_2[X_2 != "United-States"]
X_2 = X_2[X_2 != "United-States"]


#%%
print("Sueldos bajos:")
for x, y in zip(X_1, Y_1):
    print("\t%s: %d, %2.2f %%" % (x, y, (y/(sum(Y_1))*100)))

print("----------------------------------------------")
print("Sueldos altos:")
for x, y in zip(X_2, Y_2):
    print("\t%s: %d, %2.2f %%" % (x, y, (y/(sum(Y_2)))*100))


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,gridspec_kw = {'width_ratios':[2,1,1,1]})
#histograma
ax1.set_title("Frecuencia del atributo native-country por clases")
ax1.set_xlabel("Country")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X_1,Y_1,label='<=50K')
ax1.bar(X_2,Y_2,bottom=Y_1,label='>50K')

ax1.legend()

for i, v in enumerate(Y_1+Y_2):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de native-country en <=50k")
patches, texts, _ = ax2.pie(Y_1,autopct="%2.1f")
ax2.axis('equal')

ax3.set_title("Distribución de native-country en >50k")
ax3.pie(Y_2,autopct="%2.1f")
ax3.axis('equal')
    
ax4.legend(patches,X_1,loc="right")
ax4.axis('off')
    
plt.show()

#%% [markdown]
# ## Clase

#%%
X,Y=np.unique(data.CLASS,return_counts=True)


#%%
pylab.rcParams['figure.figsize'] = (20, 5)
fig, (ax1, ax2, ax3) = plt.subplots(1,3,gridspec_kw = {'width_ratios':[2,2,1]})
#histograma
ax1.set_title("Frecuencia de clases")
ax1.set_xlabel("Clase")
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Frecuencia")
ax1.bar(X,Y)

ax1.legend()

for i, v in enumerate(Y):
    ax1.text(i-.25,v+300, str(v), color='black', fontweight='bold')
    
ax2.set_title("Distribución de clases")

patches, _, _ = ax2.pie(Y,autopct="%2.1f")
ax2.axis('equal')
    
ax3.legend(patches,X,loc="right")
ax3.axis('off')
    
plt.show()

#%% [markdown]
# Podemos ver una clara descompensación en las clases, sin embargo 7.841 instancias deberían ser suficientes para clasificar la clase `>50k`.

