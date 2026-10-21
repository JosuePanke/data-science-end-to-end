import os

import snowflake.connector

os.makedirs('./plantilla-eda/datos', exist_ok=True) # Carpeta para el CSV exportado

# Muestra aleatoria de la OBT del PSet2 (Ecobici). La tabla completa tiene demasiadas filas,
# demasiado para un CSV; 200 mil filas alcanzan para explorar.
FILAS_MUESTRA = 200000

conexion = snowflake.connector.connect(
    account=os.environ['SNOWFLAKE_ACCOUNT'],
    user=os.environ['SNOWFLAKE_USER'],
    password=os.environ['SNOWFLAKE_PASSWORD'],
    role=os.environ.get('SNOWFLAKE_ROLE') or 'ACCOUNTADMIN',
    warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE') or 'WH_PSET2',
    database=os.environ.get('SNOWFLAKE_DATABASE') or 'ECOBICI',
)

consulta = f'select * from OBT.OBT_ESTACION_VENTANA sample ({FILAS_MUESTRA} rows)'
datos = conexion.cursor().execute(consulta).fetch_pandas_all()
conexion.close()

datos.columns = [c.lower() for c in datos.columns] # Snowflake devuelve los nombres en MAYUSCULAS
datos.to_csv('./plantilla-eda/datos/obt_muestra.csv', index=False)
print('Filas exportadas:', len(datos))
