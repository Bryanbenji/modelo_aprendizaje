import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Cargar las variables de entorno
load_dotenv()

# Configuración de la base de datos desde el archivo .env
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = "mysql+pymysql://root:Udla1@localhost:3306/wst"

def load_and_process_data():
    # Conectar a la base de datos
    engine = create_engine(DATABASE_URL)
    # Consulta SQL proporcionada
    query_ticket_materiales = '''
    SELECT 
        tm.material_id AS MaterialID,
        tm.fecha,
        tm.cantidad AS Demand,
        tm.stock_anterior,
        tm.stock_actual,
        CASE 
            WHEN tm.serial IS NOT NULL THEN 'Serializado'
            ELSE 'No Serializado'
        END AS TipoMaterial,
        cm.name AS MaterialName,
        cm.description AS MaterialDescription,
        cm.stock_minimo_default AS StockMinimo,
        CASE 
            WHEN tm.serial IS NOT NULL THEN COALESCE(ms_salida.count, 0)
            ELSE COALESCE(ms_sinserial.quantity, 0)
        END AS StockActual
    FROM ticket_materiales tm
    LEFT JOIN catalogo_materiales cm 
        ON cm.codigo_equipo = tm.codigo_material
    LEFT JOIN (
        SELECT
            id AS idserial,
            idmateriales,
            COUNT(*) AS count
        FROM
            materiales_serial
        WHERE
            estado IN ('en uso', 'asignado')
        GROUP BY
            idserial
    ) ms_salida 
        ON tm.material_id = ms_salida.idserial
    LEFT JOIN (
        SELECT
            id AS idsinserial,
            idmateriales,
            SUM(quantity) AS quantity
        FROM
            materiales_sinserial
        GROUP BY
            idsinserial
    ) ms_sinserial 
        ON tm.material_id = ms_sinserial.idsinserial;
    '''

    # Cargar los datos
    df_ticket_materiales = pd.read_sql(query_ticket_materiales, engine)

    # Procesar los datos
    df_ticket_materiales['fecha'] = pd.to_datetime(df_ticket_materiales['fecha'])
    df_ticket_materiales['stock_anterior'] = df_ticket_materiales['stock_anterior'].astype(float)
    df_ticket_materiales['stock_actual'] = df_ticket_materiales['stock_actual'].astype(float)
    df_ticket_materiales['Demand'] = df_ticket_materiales['Demand'].astype(float)
    df_ticket_materiales['RealUsage'] = df_ticket_materiales['stock_anterior'] - df_ticket_materiales['stock_actual']
    df_ticket_materiales['YearMonth'] = df_ticket_materiales['fecha'].dt.to_period('M').dt.to_timestamp()

    # Agrupar por mes y material
    df_monthly_demand = df_ticket_materiales.groupby(['MaterialID', 'YearMonth'])['Demand'].sum().reset_index()
    df_real_usage = df_ticket_materiales.groupby(['MaterialID', 'YearMonth'])['RealUsage'].sum().reset_index()

    # Unir con los datos de stock
    df = pd.merge(df_monthly_demand, df_ticket_materiales[['MaterialID', 'StockActual', 'MaterialName']].drop_duplicates(), on='MaterialID')
    df = pd.merge(df, df_real_usage, on=['MaterialID', 'YearMonth'], how='left')

    

    # Preparar datos para el modelo
    df_model = df[['MaterialID', 'YearMonth', 'Demand', 'RealUsage', 'StockActual', 'MaterialName']].dropna()

    print(df_model.head())
    
    return df_model
