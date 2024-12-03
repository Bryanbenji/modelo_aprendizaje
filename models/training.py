import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from window_ops.ewm import ewm_mean
from datetime import datetime, timedelta

def train_model(df_model):
    # Obtener el mes y año actual
    current_date = datetime.now()
    validation_start_date = current_date.replace(day=1)  # Primer día del mes actual
    train_end_date = validation_start_date - timedelta(days=1)  # Último día del mes anterior

    # Convertir a cadenas en formato YYYY-MM-DD
    validation_start_date_str = validation_start_date.strftime('%Y-%m-%d')
    train_end_date_str = train_end_date.strftime('%Y-%m-%d')

    # Dividir los datos en entrenamiento y validación
    train_data = df_model[df_model['YearMonth'] < '2023-12-01']
    validation_data = df_model[df_model['YearMonth'] == '2023-12-01']

    # Validar que no estén vacíos
    if train_data.empty or validation_data.empty:
        raise ValueError("Los datos de entrenamiento o validación están vacíos. Verifica el rango de fechas o la cantidad de datos disponibles.")

    
    # Configuración del modelo
    models = [
        make_pipeline(SimpleImputer(strategy='mean'), RandomForestRegressor(random_state=0, n_estimators=100)),
        XGBRegressor(random_state=0, n_estimators=100)
    ]



    # Crear el pipeline de predicción
    model = MLForecast(
        models=models,
        freq='ME',
        lags=[2, 8],  # Puedes ajustar los lags según lo necesario
        lag_transforms={
            2: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4)],
            8: [(ewm_mean, 0.5)],
        }, 
        num_threads=6
    )

    # Entrenar el modelo
    try:
        model.fit(train_data, id_col='MaterialID', time_col='YearMonth', target_col='Demand', static_features=['MaterialID'], dropna=False )
    except ValueError as e:
        raise ValueError(f"Error al entrenar el modelo: {e}")
    
    return model




