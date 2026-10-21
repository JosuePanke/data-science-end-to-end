import pandas as pd
import os
import matplotlib.pyplot as plt

os.makedirs('./plantilla-ejercicio2/figuras', exist_ok=True) # Hacer una carpeta para guardar las imagenes

viajes = pd.read_csv('./starter/viajes_taxi.csv')

print('Analisis exploratorio de relaciones')

limpios = viajes.query('tarifa != 999.99 and duracion_min >= 0')
print('Filas originales:', len(viajes))
print('Filas sin errores:', len(limpios))

plt.figure()
plt.scatter(limpios['distancia_km'], limpios['tarifa'], s=5, alpha=0.5)
plt.title('Dispersion de distancia_km vs tarifa')
plt.xlabel('distancia_km')
plt.ylabel('tarifa (USD)')
plt.savefig('./plantilla-ejercicio2/figuras/distancia_vs_tarifa.png')
plt.close()

# Puntos que no siguen la tendencia: la linea horizontal de tarifa fija y los viajes con distancia 0

tarifa_fija = limpios.query('tarifa == 70')
print('Viajes con tarifa fija de 70:', len(tarifa_fija))
print(tarifa_fija[['zona_origen', 'zona_destino']].value_counts())
print(tarifa_fija['distancia_km'].describe())

distancia_cero = limpios.query('distancia_km == 0')
print('Viajes con distancia 0:', len(distancia_cero))
print(distancia_cero[['distancia_km', 'duracion_min', 'tarifa']].describe())

#propina promedio y viajes con propina 0 por metodo de pago

propinas = limpios.groupby('metodo_pago').agg(
    propina_promedio=('propina', 'mean'),
    viajes=('propina', 'count'),
    viajes_propina_cero=('propina', lambda p: (p == 0).sum()),
)
print(propinas.round(2))

#correlacion con la tarifa

print(limpios[['distancia_km', 'duracion_min', 'pasajeros', 'propina', 'tarifa']].corr()['tarifa'].round(3))

#pasajeros vacios y en 0

print('Viajes con pasajeros vacio:', limpios['pasajeros'].isna().sum())
print('Viajes con pasajeros en 0:', (limpios['pasajeros'] == 0).sum())
