# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

La exploración responde si los datos sirven para el problema antes de modelar. Por ejemplo qué contiene cada columna y de qué tipo es, cuántos datos faltan y dónde, si hay filas duplicadas, cómo se distribuye cada variable, si hay valores atípicos o imposibles, si las categorías están desbalanceadas y qué variables están relacionadas entre sí o con el objetivo.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una sola variable propone la estadística descriptiva (media, mediana, cuartiles, mínimo y máximo) junto con histogramas y diagramas de caja para ver la distribución. Para la relación entre dos variables propone diagramas de dispersión y la matriz de correlación cuando las dos son numéricas, y comparar el promedio por grupo cuando una de ellas es categórica.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

El modelo se desvía hacia esos valores. Con las 15 tarifas de 999.99 la media de la tarifa sube de 26.52 a 31.39 y la desviación estándar llega a 70.99, así que un modelo que minimiza el error cuadrático trata de acercarse a esos puntos y sobreestima la tarifa de viajes parecidos. Mirando solo el resultado final probablemente no lo notaría, porque el error promedio puede verse aceptable o simplemente alto sin decir por qué. Se nota al mirar los datos antes, por ejemplo el máximo de 999.99 en `describe()` o una barra aislada a la derecha del histograma.

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El máximo de la tarifa, 999.99, que está muy lejos del tercer cuartil de 32.55 y parece un valor de relleno. Y el mínimo de `duracion_min`, que es -74, porque un viaje no puede durar un tiempo negativo. También llama la atención que la distancia mínima sea 0 km.

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No. El IQR solo dice que un valor está lejos del centro, no que esté mal. De los 334 atípicos (tarifa mayor a 60.40) solo 15 son errores con 999.99. Otros 302 tienen una tarifa exacta de 70 y todos van o vienen de JFK, que es la tarifa fija del aeropuerto. Los 17 restantes son viajes largos de 19 a 24 km con tarifas de 61 a 78, que son razonables para esa distancia. Para averiguarlo filtré los atípicos, los agrupé por valor de tarifa y por zona de origen y destino, y revisé si la distancia explicaba la tarifa.

**La formulación completa, tal como la resuelvo yo:**

El script [`starter/EDA.py`](./starter/EDA.py) se corre con `docker compose run --rm --build python-app` y guarda las figuras en `data-contenedor/`. Primero imprime `info()`, los tipos, los faltantes, los duplicados y `describe()`. Después grafica el histograma y el diagrama de caja de la tarifa, calcula los límites del IQR y separa los atípicos. Al final grafica la dispersión de tarifa contra distancia e imprime la matriz de correlación.

Lo que encontró: 3000 filas y 10 columnas sin duplicados. Faltan 120 valores de `pasajeros` y 60 de `metodo_pago`. La tarifa tiene una mediana de 22.60 pero una media de 31.39 por los 15 valores de 999.99. Hay 10 viajes con duración negativa y 25 con distancia 0. Con Q1 de 13.99 y Q3 de 32.55 el límite superior del IQR queda en 60.40 y marca 334 atípicos, de los cuales solo los 15 de 999.99 son errores. La mayoría son los 302 viajes de JFK con tarifa fija de 70.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

Se corre con `docker compose run --rm --build ejercicio2` (las figuras quedan en `data-contenedor/ejercicio2/`). Al quitar las 15 tarifas de 999.99 y las 10 duraciones negativas quedan 2975 de 3000 viajes.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

La tendencia es lineal (correlación 0.96), pero hay dos grupos que no la siguen. El primero es una línea horizontal en 70 USD: **300 viajes** con tarifa fija de 70, todos con origen o destino en JFK Airport y distancias de 0 a 45 km. Es la tarifa plana del aeropuerto, no un error. El segundo son **25 viajes con distancia 0** y tarifas de 7 a 70 USD (duración media de 27 min), probablemente fallos del GPS o del taxímetro. Esos 300 viajes de JFK son casi todos los atípicos que el IQR marcó en el Ejercicio 1, así que no hay que quitarlos: el modelo necesita una variable que diga si el viaje es de JFK.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

En efectivo la propina promedio es 0.00 y los **888 de 888** viajes tienen propina 0. Con tarjeta el promedio es 4.73 y ningún viaje (0 de 2027) tiene propina 0. Esto describe al sistema, no a los pasajeros: la propina en efectivo no se registra. Si se modela la propina, `propina = 0` en efectivo significa "dato faltante", no "no dejó propina".

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La correlación de la duración con la tarifa es 0.956, casi igual a la de la distancia (0.960), pero la duración solo se conoce cuando el viaje termina. Usarla sería fuga de información: el modelo daría buenos resultados en el entrenamiento y fallaría en producción. Lo mismo vale para la propina (0.633). Las variables válidas son las que se conocen al pedir el taxi: distancia estimada, zonas, hora y pasajeros (aunque pasajeros casi no correlaciona, -0.051, y tiene **119 vacíos y 19 en 0**).

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

Mi proyecto es el PSet2 Ecobici: el archivo listo para modelar es la OBT `ECOBICI.OBT.OBT_ESTACION_VENTANA` (una fila por estación × día × franja de 15 min, 23.9 M filas) y el objetivo es `y_salidas_prox_60m`. Como la tabla no cabe en un CSV, exporto una muestra aleatoria de 200 000 filas:

```
docker compose run --rm --build exportar-obt   # lee las credenciales del .env del PSet2
docker compose run --rm ejercicio3             # figuras en data-contenedor/ejercicio3/
```

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

La OBT se reconstruyó con un solo mes, agosto 2026. Son 1 473 683 viajes que quedan en 2 008 800 filas de estación × día × franja para 678 estaciones. La muestra tiene **200 000 filas y 48 columnas**.

**Faltantes:** las 9 columnas de disponibilidad GBFS (`bicis_promedio`, `pct_tiempo_vacia`, `n_snapshots` y las demás) están vacías en el **100 %** de las filas. `salidas_mismo_momento_semana_pasada` falta en 45 183 filas (22.6 %), `salidas_mismo_momento_ayer` en 6 427, los rezagos de 15 min y de la última hora en 52, y el objetivo en 71.

**Objetivo `y_salidas_prox_60m`:** Q1 = 0, Q3 = 4, IQR = 4 y límite superior = 10. **El IQR marca 10 569 atípicos (5.3 %)**. La distribución tiene una cola larga a la derecha y un máximo de 168 salidas en una hora (`figuras/distribucion_objetivo.png`).

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Hice dos cosas. La primera es una dispersión de `salidas_ultima_hora` contra el objetivo (`ultima_hora_vs_objetivo.png`): la relación es positiva pero muy dispersa (correlación 0.637), con una nube densa entre 0 y 25. La segunda es el promedio del objetivo por hora, separando días laborales de fines de semana (`objetivo_por_hora.png`). Los días laborales tienen **dos picos**, a las 7 h (6.37 salidas) y a las 17 h (6.49), que corresponden a ir y volver del trabajo. Los fines de semana tienen **un solo pico** a las 11 h (4.96). De 1 a 3 h el promedio es 0. La hora sola casi no correlaciona con el objetivo (0.161) porque la relación no es lineal: depende del tipo de día.

Las correlaciones con el objetivo que confirman lo anterior son `salidas_ultima_hora` 0.637, `salidas` 0.578, `salidas_lag_15m` 0.546, `salidas_mismo_momento_semana_pasada` 0.536 y `salidas_mismo_momento_ayer` 0.440.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1:**

**Las 9 columnas de disponibilidad GBFS están vacías en el 100 % de las filas** (`n_snapshots` no nulo en 0 de 200 000). El feed GBFS no tiene histórico: solo existe desde que el colector empezó a correr (octubre 2026), y los viajes cargados son de agosto. **Qué cambia:** esas columnas no se pueden usar para entrenar, y tampoco puedo medir el problema de censura, es decir, que `salidas = 0` en una estación vacía signifique "no había bicis" y no "no hubo demanda". Para usarlas hay que entrenar con meses en los que el colector ya estuviera corriendo.

**Hallazgo 2:**

**El objetivo es un conteo con muchos ceros y una cola larga.** El 36.3 % de las franjas tiene `y_salidas_prox_60m = 0` (72 554 de 199 929), la mediana es baja y el IQR marca 10 569 atípicos (más de 10 salidas). Esos atípicos no son errores: se concentran en estaciones muy activas en hora pico (por ejemplo, la estación `028` tiene 17 a 28 salidas por hora). **Qué cambia:** no quito los atípicos, porque son justo los casos que el rebalanceo necesita anticipar. Uso un modelo de conteo (regresión Poisson o un GBM con pérdida Poisson) o transformo con `log1p`, y agrego la interacción `hora × es_dia_laboral`, porque el patrón de picos cambia según el tipo de día.

**Hallazgo 3:**

**Hay columnas del futuro dentro de la OBT y rezagos vacíos al inicio del periodo.** `y_salidas_prox_30m` (correlación 0.894), `y_llegadas_prox_60m` (0.658), `y_llegadas_prox_30m` y `y_flujo_neto_prox_60m` se calculan con lo que pasa *después* de la franja. Si entran como variables, el modelo hace trampa (fuga de información, igual que la duración en el caso de los taxis). Además, `salidas_mismo_momento_semana_pasada` está vacía en el 22.6 % de las filas (la primera semana de agosto no tiene semana anterior) y el objetivo falta en 71 filas (el final del mes). **Qué cambia:** excluyo todas las columnas `y_*` menos el objetivo, y también `construido_en`. Quito las 71 filas sin objetivo y cargo al menos un mes previo para que los rezagos semanales no dejen fuera la primera semana.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Las crearía a partir de columnas que ya existen. Para JFK haría una variable `es_jfk` que valga 1 cuando `zona_origen` o `zona_destino` sea JFK Airport, así el modelo puede aprender que esos viajes cuestan 70 sin importar la distancia. Para la hora sacaría la hora de `fecha_hora_inicio` con pandas y la usaría directa o agrupada en franjas como madrugada, hora pico y noche. También podría agregar el día de la semana o si es fin de semana.
