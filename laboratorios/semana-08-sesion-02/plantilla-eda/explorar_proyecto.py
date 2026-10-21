import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./plantilla-eda/figuras', exist_ok=True) # Hacer una carpeta para guardar las imagenes

# Muestra de la OBT del PSet2 (Ecobici): una fila por estacion x dia x franja de 15 minutos.
# Se genera con exportar_obt.py.
datos = pd.read_csv('./plantilla-eda/datos/obt_muestra.csv')

# Variable objetivo: salidas de bicicletas de la estacion en los proximos 60 minutos.
# Aqui solo se explora; el modelo que la predice se entrena en sesiones posteriores.
OBJETIVO = 'y_salidas_prox_60m'

print('Analisis exploratorio del proyecto')

# resumen

print('Filas y columnas:', datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# distribucion de la variable objetivo

plt.figure()
plt.hist(datos[OBJETIVO].dropna(), bins=40)
plt.title('Distribucion de ' + OBJETIVO)
plt.xlabel('salidas en los proximos 60 min')
plt.ylabel('Numero de franjas')
plt.savefig('./plantilla-eda/figuras/distribucion_objetivo.png')
plt.close()

# atipicos de la variable objetivo con la regla del IQR

q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)

iqr = q3 - q1

limite_superior = q3 + 1.5 * iqr
limite_inferior = q1 - 1.5 * iqr

atipicos = datos.query(f'{OBJETIVO} < {limite_inferior} or {OBJETIVO} > {limite_superior}')
print('Q1:', q1, 'Q3:', q3, 'IQR:', iqr, 'Limite superior:', limite_superior)
print(f'Atipicos de {OBJETIVO} segun IQR:', len(atipicos))
print(atipicos[['estacion_codigo', 'fecha', 'etiqueta', 'salidas_ultima_hora', OBJETIVO]].head(10))

#  relacion del objetivo con la variable que mas deberia influir (salidas de la ultima hora)

plt.figure()
plt.scatter(datos['salidas_ultima_hora'], datos[OBJETIVO], s=3, alpha=0.3)
plt.title('Dispersion de salidas_ultima_hora vs ' + OBJETIVO)
plt.xlabel('salidas_ultima_hora')
plt.ylabel(OBJETIVO)
plt.savefig('./plantilla-eda/figuras/ultima_hora_vs_objetivo.png')
plt.close()

# Promedio del objetivo por hora del dia, separando dias laborales y fines de semana

por_hora = datos.groupby(['hora', 'es_dia_laboral'])[OBJETIVO].mean().unstack()
print(por_hora.round(2))

plt.figure()
plt.plot(por_hora.index, por_hora[True], label='dia laboral')
plt.plot(por_hora.index, por_hora[False], label='fin de semana / feriado')
plt.title('Promedio de ' + OBJETIVO + ' por hora')
plt.xlabel('hora')
plt.ylabel(OBJETIVO)
plt.legend()
plt.savefig('./plantilla-eda/figuras/objetivo_por_hora.png')
plt.close()

# consultas para confirmar los hallazgos

# cuantas franjas tienen objetivo 0 (ceros que dominan la distribucion)
print('Franjas con objetivo 0:', (datos[OBJETIVO] == 0).sum(), 'de', datos[OBJETIVO].notna().sum())
print('Porcentaje con objetivo 0:', round(100 * (datos[OBJETIVO] == 0).mean(), 2))

# faltantes de rezagos y objetivos (bordes de la ventana) y de disponibilidad GBFS
print('Filas con objetivo vacio:', datos[OBJETIVO].isna().sum())
print('Filas sin rezago de la semana pasada:', datos['salidas_mismo_momento_semana_pasada'].isna().sum())
print('Porcentaje de filas con disponibilidad GBFS:', round(100 * datos['n_snapshots'].notna().mean(), 2))
print(datos.groupby(datos['n_snapshots'].notna())['fecha'].agg(['min', 'max', 'count']))

# estaciones vacias, donde salidas = 0 puede ser falta de bicis y no falta de demanda
con_gbfs = datos.query('n_snapshots == n_snapshots') # filas con disponibilidad
con_gbfs = con_gbfs.assign(vacia_mitad=con_gbfs['pct_tiempo_vacia'] >= 0.5)
print(con_gbfs.groupby('vacia_mitad')[OBJETIVO].agg(['mean', 'count']).round(2))

# Correlaciones con el objetivo
columnas = [
    'salidas', 'llegadas', 'salidas_lag_15m', 'salidas_ultima_hora', 'llegadas_ultima_hora',
    'salidas_mismo_momento_ayer', 'salidas_mismo_momento_semana_pasada',
    'capacidad_actual', 'bicis_promedio', 'pct_tiempo_vacia', 'hora',
    'y_salidas_prox_30m', 'y_llegadas_prox_60m', OBJETIVO,
]
print(datos[columnas].corr()[OBJETIVO].sort_values(ascending=False).round(3))

# Estaciones que ya no estan en el catalogo actual
print(datos.groupby('en_catalogo_actual')[OBJETIVO].agg(['mean', 'count']).round(2))
