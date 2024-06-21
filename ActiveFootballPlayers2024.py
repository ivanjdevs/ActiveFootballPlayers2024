#!/usr/bin/env python
# coding: utf-8

# <h1 align=right><font size = 2>Author: ivanjdevs</font></h1>
# <h1 align=right><font size = 2>Created: 17-May-2024</font></h1>

# <h1 align=center><font size = 5>Active football players</font></h1>

# In[1]:


# Importamos primero los módulos y librerías a usar (al menos las que creemos que necesitamos de momento):
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


# In[2]:


#El archivo que contiene los datos es tipo .csv; leámos dicho archivo y guardemos su contenido en una variable llamada dataf:
dataf=pd.read_csv("football_players.csv")


# In[3]:


dataf.head(20)


# ##### Primero, obtener una idea general del dataset: cuántas filas y columnas hay y qué tipos de datos que almacena cada columna.

# In[4]:


# Cantidad de filas y columnas:
dataf.shape


# In[5]:


# Verifiquemos los tipos de datos de cada columna:
dataf.dtypes


# ##### Las dos primeras columnas no las vamos a usar, luego, descartarlas de una vez.

# In[6]:


dataf.drop(['nationality','image'], axis=1, inplace=True)


# In[7]:


dataf.head(20)


# <h4> Primero identificar si hay valores nulos (NaN) </h4>
# <h5> Una primera aproximación es usando el método info, que presenta, para cada columna, el tipo de dato y la cantidad de valores no nulos en cada una. </h5>

# In[8]:


dataf.info()


# In[9]:


# Verbose indicates whether to print the full summary of the DataFrame or not.
dataf.info(verbose=False)


# In[10]:


# Otra forma de validar si hay datos NaN:
dataf.isnull().sum()


# <h5> El método shape ejecutado mas arriba nos reportó que el dataframe tiene 585 filas. El método info nos acaba de arrojar que en cada columna hay 565 valores no nulos,
# luego nuestro dataframe no tiene registros NaN. Pero eso no quiere decir que no hayan valores no válidos de otro tipo, como caracteres tipo '?', '#', '!', '@', '/', '', '%'
# Revisemos si se tienen caracteres de ese tipo dentro del dataframe. </h5>

# In[11]:


# Esto nos arrojará un array donde nos dirá si cada caracter dentro de la lista 'carac' existe (TRUE) o no (FALSE) dentro del dataframe:
carac=['?', '#', '!', '@', '/', '', '%']
[True if item in dataf.values else False for item in carac]


# <h5>Perfecto, no hay caracteres de ese tipo en el dataframe.  </h5>

# <h5>Habiendo validado que no hay datos nulos ni no válidos y, observando que cada columna tiene el dato apropiado (object, int, etc), podemos
# pasar a revisar si hay datos duplicados.</h5>

# In[12]:


# Revisemos cuantos valores únicos hay, para validar si hay filas duplicadas. En este caso, nos interesa saber si hay jugadores repetidos.
dataf['name'].nunique()


# <h5> Sabemos que el dataframe tiene 565 filas, y la línea anterior nos dice que hay 380 registros distintos para la columna name, lo cual significa que 
# tenemos 185 (565-380) valores repetidos. Averigüemos mas sobre ello. </h5>

# <h4> Usemos el método duplicated </h4>

# <h5>
# El método duplicated de la clase DataFrame permite identificar filas repetidas.
# Como resultado se obtiene una serie cuyos valores son de tipo bool, cada uno de los cuales está asociado a una fila.
# El valor True indica que ya existe una fila anterior en el dataframe con los mismos valores (la primera aparición no se considera repetida).
# En caso contrario, el valor será False.
# La forma más básica de aplicar el método duplicated es la siguiente:</h5>

# In[13]:


dataf.duplicated()


# <h5> Lo anterior no es muy útil para dataframes largos y medianamente largos como el que estamos trabajando, ya que el resultado que nos arroja es una lista resumida
# y no podemos visualizar cuales valores son True, es decir, cuáles filas específicamente son las repetidas. Pero sigamos usando más opciones del método duplicated.</h5>

# In[14]:


# Comprobemos que en realidad como lo establecimos anteriormente, tenemos 185 filas repetidas.
dataf.duplicated().sum()
# dataf['name'].duplicated().sum()  ##Obtener solo los duplicados de la serie 'name'


# <h5>Efectivamente tenemos 185 filas exactamnte iguales, es decir, no solo se repite el nombre del jugador sino también su club y su posición, 
# atributos que podrían ser diferentes, sabiendo que un jugador puede cambiar de club y de posición.</h5>

# In[15]:


# Revisemos ahora las filas repetidas mediante un filtro:
#dataf[filtro]
dataf[dataf.duplicated()]


# <h5>Igual que cuando ejecutamos la línea dataf.duplicated(), el filtro anterior nos presenta solo un resumen del total de las 185 filas repetidas,
# pero al menos obtenemos un resultado mas visual, logrando identificar algunos de los jugadores repetidos. </h5>

# <h5>Habiendo identificado que tenemos 185 filas exactamente iguales, podemos optar por descartarlas entonces. </h5>

# In[16]:


# Método drop_duplicates:
dataf.drop_duplicates(inplace=True)
dataf


# <h5>Podemos pasar a ejecutar un análisis exploratorio de datos.</h5>

# <h3>EXPLORATORY DATA ANALYSIS</h3>

# <h5>Antes de pasar propiamente a hacer una exploración del dataframe, recordemos algunos métodos para obtener datos de una columna, de varias, 
# y como la sintaxis hace la diferencia entre obtener una serie o un dataframe.</h5>

# In[17]:


#Obtener todos los datos contenidos en una columna.

##Esta sintaxis nos arroja como resultado una serie:
dataf.name


# In[18]:


# Esta sintaxis también nos arroja como resultado una serie:
dataf['name']


# In[19]:


# Esta sintaxis nos arroja como resultado un dataframe (obsérvese como el resultado se presenta en una forma diferente a las dos anteriores):
dataf[['name']]


# In[20]:


# Obtener las últimas diez filas del dataframe:
dataf.tail(10)


# In[21]:


# Elegir una fila al azar:
dataf.sample()


# In[22]:


# Elegir varias filas al azar:
dataf.sample(5)


# <h4> --------------Now, let's get some basics statistics of the dataframe.-------------------- </h4>

# In[23]:


# El método describe, adicionando el parámetro include='all', ofrece información estadística básica de TODAS las columnas:
dataf.describe(include='all') 


# In[24]:


# Dejando el método describe sin especificar ningún parámetro interno, se obtienen estadísticas solo de las columnas con datos numéricos.
dataf.describe()


# <h5> Ya acá podemos ver datos como </h5>
# <ul>  
# <li>El jugador o jugadores mas jóvenes tienen 19 años.</li>
# <li>Los jugadores de más edad tienen 39 años.</li>
# <li>La edad promedio de los jugadores de la lista es de 28 años.</li>
# <li>El rango de estatura está entre los 1.65 m y casi 2 m.</li>
# <ul>

# ##### --------------Hagamos unos filtros--------------------

# In[25]:


# Veamos quiénes son los jugadores de mas edad:
dataf[dataf['age']==39]


# In[26]:


# Veamos quiénes son los jugadores más jovenes:
dataf[dataf['age']==19]


# In[27]:


# ¿Que posiciones están listadas en la columna position?
dataf['position'].unique()


# In[28]:


## ¿Cuantos registros tiene la lista anterior? (Pa no contar hey (ya conté y hay 22))
dataf['position'].nunique()


# In[29]:


# ¿Cuántas veces aparece cada posición en el dataframe?
dataf['position'].value_counts()


# In[30]:


# En la lista anterior se observa que la posición más común dentro de los jugadores listados es 'Defender Centre'. Se puede validar lo anterior también así:
dataf['position'].value_counts().idxmax()


# In[31]:


# Todos los clubes listados:
dataf['club'].unique()


# In[32]:


# ¿Cuántos equipos hay en el dataframe (ya los conté en el array anterior, hay 70)
dataf['club'].nunique()


# In[33]:


# Los diez equipos con mas apariciones en el dataframe:
dataf['club'].value_counts().head(10)


# In[34]:


# Filtro. Veamos, por ejemplo, cuáles son los 15 jugadores del Liverpool que el dataframe cita:
dataf[dataf['club']=='Liverpool']


# In[35]:


# Si solo quisiera los nombres de los jugadores mas no todas las demás columnas, hago lo siguiente:
dataf[dataf['club']=='Liverpool']['name']


# In[38]:


# ¿Cuántos jugadores son left-footed, right-footed y cuantos ambidiestros?
dataf['foot'].value_counts()


# <h4> Hagamos una maniobra agrupativa bacana. Listar las posiciones del dataframe (cosa que ya se hizo antes) y al frente presentar la edad promedio de todos los 
# jugadores que juegan en dicha posición.</h4>

# In[39]:


dataf.groupby(['position']).mean()['age']


# In[40]:


# O mesmo anterior mais agora con la altura dos jogadores:
dataf.groupby(['position']).mean()['height'].round(2)


# <h5>De lo anterior vemos que los jugadores más altos son los goalkeepers, seguidos por los defensas centrales y los delanteros.</h5>

# In[41]:


# Edad promedio y altura promedio del top 10 de los equipos del dataframe #

##Primero, filtrar el dataframe para hallar los 10 equipos con mas apariciones en el dataframe (eso ya lo hicimos antes):
data10=dataf['club'].value_counts().head(10)
data10


# In[42]:


# El método Index aplicado a la serie anterior nos devuelve los registros localizados en la primera 'columna':
data10.index


# In[43]:


# Ahora filtramos el dataframe original para obtener solo las filas donde aparecen los equipos de la anterior lista. Esto lo guardamos en un nuevo dataframe dataf1.
dataf1=dataf[dataf['club'].isin(data10.index)]
dataf1


# In[44]:


# Ahora, en el dataframe dataf1, hacemos una maniobra agrupativa.
# Agrupamos por club, y al frente de cada uno listamos el promedio de la edad y el promedio del altura de sus jugadores:
dataf1.groupby(['club']).mean()[['age','height']]


# <h5> Del listado de los 10 equipos con más apariciones en el dataframe, se observa que:</h5>
# <ul>  
# <li>El club con menor edad promedio entre sus jugadores es el Arsenal</li>
# <li>El club con mayor promedio de edad es el Inter</li>
# <li>El club con mayor promedio de altura entre sus jugadores es el LIVERPOOL y el Man. Utd</li>
# <li>El club con menor promedio de altura es el Barcelona</li>
# <ul>

# In[45]:


# Guardemos el resultado anterior en un nuevo dataframe:
d10stats=dataf1.groupby(['club']).mean()[['age','height']]
d10stats


# In[46]:


# Quitemos ese aparente doble nivel en los headers:
d10stats.index.name = None
d10stats


# In[47]:


# En el dataframe d10stats, la columna donde están listados los clubes es el actual indice, lo cual es bastante útil. 
# Pero si se quisiera colocar un nuevo índice, esto es, una nueva columna al inicio que le asignara un número a cada fila, podemos hacer lo siguiente:

d10stats.reset_index(inplace = True)
d10stats


# In[48]:


# Cambiemos el nombre de la segunda columna:
d10stats.rename(columns={'index':'club'}, inplace=True)


# In[49]:


d10stats


# <h5>----Hagamos unas maniobras pero con las columnas tipo texto------------</h5>

# In[50]:


import re ## importar la librería re (Regular Expressions) Una expresión regular es una cadena de texto que conforma un patrón de búsqueda.

# Hállate ahí los nombres de los jugadores que empiezan con A:
dataf.loc[dataf['name'].str.contains('^a[a-z]*', flags=re.I, regex=True)]


# In[51]:


## Hállate ahí los nombres de los clubes que empiezan con la letra V:
dataf.loc[dataf['club'].str.contains('^v[a-z]*', flags=re.I, regex=True)]


# In[55]:


# Hállate ahí si hay nombres que empiezan por Lio (ya sabes a quién estamos buscando, eh ¿?):
dataf.loc[dataf['name'].str.contains('^lio[a-z]*', flags=re.I, regex=True)]


# <h3>DATA VISUALIZATION</h3>

# In[48]:


# Vamos sacando una foto. De la librería sns, aplicate ahí la gráfica distplot:
sns.distplot(dataf['age'])


# ##### Se observa que la mayoría de los jugadores están concentrados alrededor de los 25 a 30 años.

# In[49]:


# Observemos como se distribuyen las edades en los 10 equipos con mas apariciones en el dataframe.

f, ax=plt.subplots(figsize=(8,6)) #definir el área del gráfico o pedazo de pantalla que reservo para la gráfica (de la librería matplotlib vas a agarrar subplots)
fig=sns.boxplot(x='club', y='age', data=dataf1)  # definir la propia gráfica
fig.axis(ymin=15, ymax=45) #modificando el atributo axis de la figura definida fig
plt.xticks(rotation=90)  #rotate ahí las etiquetas del eje x


# <h5>El anterior gráfico boxplot reporta que:</h5>
# <ul>  
# <li>Los equipos cuyo espectro o rango de edades es mas compacto son el M. Utd el Atl. Madrid y el Arsenal</li>
# <li>Los equipos con edades mas dispersas entre sus jugadores son el Real Madrid y el Barcelona.</li>
# <li>Tenemos unos outliers en algunos clubes (PSG, Bayer M., Atl. Madrid y Arsenal)</li>
# </ul>

# In[50]:


# Comprobemos numéricamente la segunda observación. Sabemos que el dataframe dataf1 contiene información sobre esos 10 clubes. Hagamos una agrupación por club y hallemos
# la desviación estándar de la edad para cada club:
dataf1.groupby('club').std()['age']


# ##### Efectivamente se observa que donde hay mas dispersión de los datos (desviación estándar más alta) es en los clubes Real Madrid y Barcelona.

# #### Exploremos un poco el gráfico boxplot.

# In[51]:


# Traigamos la columna edad del dataframe data1, pero solo para el equipo Bayern München
# Sintáxis:  dataf1[filtro].column

dataf1[dataf1['club']=='Bayern München'].age
# dataf1[dataf1['club']=='Bayern München']['age'] otra foram equivalente a la sintaxis anterior


# In[52]:


# Hallemos los cuartiles de la lista de datos anterior:

# Filtro: dataf[dataf['club']=='Liverpool']

q75, q25 = np.percentile(dataf1[dataf1['club']=='Bayern München'].age, [75 ,25])
iqr = q75 - q25

#display interquartile range 
print("El percentil 25 es (o primer cuartil): ",q25, "; el percentil 75 es (o tercer cuartil): ", q75, " y el intercuartil es: ", iqr)


# <h5>En el gráfico boxplot vimos que algunos clubes tiene unos valores que se alejan de los demás. Se distancian bastante del grupo donde se concentran la mayoría de los datos.
# Son datos que están fuera de rango y sobresalen de los demás. En la gráfica de boxplot, se identifican visualmente como aquellos puntos con forma de diamante.</h5>
# 
# <h5>Se conocen también como valores atípicos.</h5>
# 
# <h5>El método más sencillo para hallar los valores atípicos es el test de Tukey, que toma como referencia la diferencia entre el primer cuartil (Q1) y el tercer cuartil (Q3),
# o rango intercuartílico (iqr). En un diagrama de caja se considera un valor atípico el que se encuentra 1,5 veces esa distancia de uno de esos cuartiles (atípico leve)
# o a 3 veces esa distancia (atípico extremo).</h5>
# 

# In[53]:


# De acuerdo a lo anterior, hallemos cuáles son esos límites por fuera de los cuales se considera que un valor es atípico para el club Bayern:
print("Valores límite por fuera de los cuales se considera son valores outliers: ",q25-1.5*iqr, "y", q75+1.5*iqr)


# <h5>Ayudados con el gráfico bloxplot y la lista de las edades para el Bayern, vemos que el valor atípico es 38 años.
# Pero, para el caso donde se tenga un conjunto de datos muy grande, ¿cómo hallar esto con una línea de código?</h5>

# In[54]:


##Primero, traigamos la columna edad para el equipo Bayern:
dataBM=dataf1[dataf1['club']=='Bayern München'].age
dataBM


# In[55]:


## Como lo anterior es una serie, podemos acceder a los objetos que contienen los índices y los valores a través de los atributos index y values de la serie.
## En este caso necesitamos son los valores
dataBM.values


# In[56]:


# Hallemos ahora los valores lejanos con un filtro y con la teoría ya explicada, es decir, los valores que están 1,5 la distancia intercuartil de los cuartiles Q1 y Q3
outliers = dataBM[(dataBM.values <(q25-1.5*iqr)) | (dataBM.values>(q75+1.5*iqr))]
outliers


# <h5>Efectivamente acabamos de comprobar que la edad considerada más alejada del grupo de datos es 38 años.
# Nos informa además que corresponde al registro con índice número 40. Si quisiéramos saber todos los datos de ese registro con índice número 40,
#  es decir, quién es ese jugador, y demás atributos, ejecutamos lo siguiente:</h5>

# In[57]:


# Primero, guardamos en un dataframe todas la filas donde aparece el equipo Bayern (es decir, hacemos un filtrado):
dataBM1=dataf1[dataf1['club']=='Bayern München']
dataBM1


# In[58]:


out = dataBM1[(dataBM1['age']<(q25-1.5*iqr)) | (dataBM1['age']>(q75+1.5*iqr))]
out


# In[59]:


#¿Cómo están distribuidas las alturas de los jugadores?
sns.distplot(dataf['height'])


# In[60]:


# Llevemos lo anterior a un histograma:

count, bin_edges = np.histogram(dataf1.height)
print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins


# In[61]:


# Ejecutemos la propia gráfica:
count, bin_edges = np.histogram(dataf.height)
dataf.plot.hist(figsize=(8, 5), xticks= bin_edges)
plt.title('Histogram for the height') # add a title to the histogram
plt.ylabel('Frequency') # add y-label
plt.xlabel('Height') # add x-label
plt.ticklabel_format(style='plain', axis='both', scilimits=(0,0))
plt.xlim([160, 205])
plt.xticks(rotation=90)
plt.show()


# In[62]:


# Traigamos un conteo que ya hicimos varias varias varias líneas arriba (línea de código 35 o algo así):
dataf['foot'].value_counts()


# In[63]:


# Grafiquemos ello:

group = ['Right', 'Left', 'Both']
plt.bar(group, dataf['foot'].value_counts()) 
plt.xlabel("Foot")
plt.ylabel("Count")
plt.title("Type of skill")


# In[64]:


# ¿Y si quisiera hacer lo anterior para un club en específico? Breve. Píllala...
group = ['Right', 'Left', 'Both']
plt.bar(group, dataf1[dataf1['club']=='Bayern München'].foot.value_counts(), color='lightblue') #---------------ACÁ SE HA HECHO UN FILTRO BACANO-----------------------------#
plt.xlabel("Foot")
plt.ylabel("Count")
##plt.title("")


# In[65]:


# El mismo anterior, solo para obtener el conteo mediante lista y no gráfico:
dataf1[dataf1['club']=='Bayern München'].foot.value_counts()


# In[68]:


## Grafiquemos lo anterior pero para vrios equipos.

fig = plt.figure() # create figure

ax0 = fig.add_subplot(2, 2, 1) # add subplot 1 (1 row, 1 columns, first plot)
ax1 = fig.add_subplot(2, 2, 2) # add subplot 2 (1 row, 2 columns, second plot)
ax2 = fig.add_subplot(2, 2, 3) # add subplot 2 (2 row, 1 columns, second plot)
ax3 = fig.add_subplot(2, 2, 4) # add subplot 2 (2 row, 2 columns, second plot) 

#xlabels = ['Right', 'Left', 'Both']
# Subplot 1: Box plot
dataf1[dataf1['club']=='Liverpool'].foot.value_counts().plot(kind='bar', color='red', figsize=(15, 8), ax=ax0) # add to subplot 1
ax0.set_ylabel('Count')
ax0.tick_params(axis='x', rotation=45)
#ax0.set_xticklabels(xlabels, rotation=0)

# Subplot 2: Bar plot
dataf1[dataf1['club']=='Bayern München'].foot.value_counts().plot(kind='bar', color='darkred', figsize=(15, 8), ax=ax1) # add to subplot 2
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
#ax1.set_xticklabels(xlabels, rotation=0)

# Subplot 3: Bar plot
dataf1[dataf1['club']=='Real Madrid'].foot.value_counts().plot(kind='bar', color='purple', figsize=(15, 8), ax=ax2) # add to subplot 2
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)
#ax2.set_xticklabels(xlabels, rotation=0)

# Subplot 2: Bar plot
dataf1[dataf1['club']=='Internazionale'].foot.value_counts().plot(kind='bar', color='darkblue', figsize=(15, 8), ax=ax3) # add to subplot 2
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)
#ax3.set_xticklabels(xlabels, rotation=0)

plt.show()


# ## Author
# <a href="https://www.linkedin.com/in/iv%C3%A1n-pinilla-%C3%A1vila-21bb45121/" target="_blank"><font size = 4>Iván P. </a>
