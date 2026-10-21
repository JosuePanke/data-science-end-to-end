import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./starter/figuras', exist_ok=True) # Hacer una carpeta para guardar las imagenes

data = pd.read_csv('./starter/viajes_taxi.csv')

print('Analisis exploratorio descriptivo')

data.info( ) # informacion de la estructura del dataframe

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())
print(data.describe())

plt.figure()
plt.hist(data['tarifa'], bins=50)
plt.title('Distribucion de la tarifa')
plt.xlabel('USD')
plt.ylabel('Numero de viajes')
plt.savefig('./starter/figuras/hist_tarifa.png')
plt.close()



plt.figure()
plt.boxplot(data['tarifa'])
plt.title('Boxplot de la tarifa')
plt.savefig('./starter/figuras/boxplot_tarifa.png')
plt.close()

# Extracion de valores atipicos

q1 = data['tarifa'].quantile(0.25)
q3 = data['tarifa'].quantile(0.75)

iqr = q3 - q1

limite_superior = q3 + 1.5 * iqr
limite_inferior = q1 - 1.5 * iqr

atipicos = data.query(f'tarifa < {limite_inferior} or tarifa > {limite_superior}')
sin_atipicos = data.query(f'tarifa >= {limite_inferior} and tarifa <= {limite_superior}')
print(atipicos)

plt.figure()
plt.scatter(sin_atipicos['tarifa'], sin_atipicos['distancia_km'])
plt.title('Dispercion de tarifa vs distancia_km')
plt.savefig('./starter/figuras/dispercion_distancia_vs_tarifa_sin_atipicos.png')
plt.close()

print(data.corr(numeric_only=True))