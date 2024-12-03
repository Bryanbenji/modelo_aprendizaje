from fastapi import FastAPI, HTTPException
import joblib
from models.training import train_model
from data.data_loader import load_and_process_data
from datetime import datetime
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

app = FastAPI()

# Cargar el modelo entrenado al iniciar el servidor
try:
    model = joblib.load("models/model.pkl")
except FileNotFoundError:
    model = None

@app.get("/")
def read_root():
    try:
        # Carga y procesa los datos
        df_train = load_and_process_data()

        # Convierte el DataFrame a un formato serializable (lista de diccionarios)
        df_as_dict = df_train.to_dict(orient="records")

        return {
            "message": "API para predicción de demanda y stock de materiales",
            "data": df_as_dict
        }
    except Exception as e:
        # Maneja errores y devuelve un mensaje informativo
        return {"error": str(e)}


@app.post("/train/")
def train():
    global model
    # Cargar y procesar datos, luego entrenar
    df_train = load_and_process_data()
    # Retirar `MaterialName` antes de entrenar
    train_data = df_train.drop(columns=["MaterialName"])
    model = train_model(train_data)
    print(model)
    joblib.dump(model, "models/model.pkl")
    return {"message": "Modelo entrenado y guardado exitosamente"}


@app.get("/predict/")
def predict(material: str, horizon: int = 3):
    
    model = joblib.load("models/model.pkl")

    if model is None:
        raise HTTPException(status_code=400, detail="El modelo no está entrenado. Usa el endpoint '/train/' primero.")
    
    # Cargar y procesar datos
    df_data = load_and_process_data()

    # Verificar columnas
    if 'RealUsage' not in df_data.columns or 'StockActual' not in df_data.columns:
        raise HTTPException(status_code=500, detail="Las columnas 'RealUsage' y 'StockActual' no están en los datos.")
    
    # Filtrar datos para el material solicitado
    material_data = df_data[df_data["MaterialName"] == material]
    
    if material_data.empty:
        raise HTTPException(status_code=404, detail=f"Material '{material}' no encontrado.")
    
    # Manejar valores nulos
    material_data = material_data.fillna(0)

    # Verificar columnas después de filtrar
    print(material_data.columns)

    # Generar predicciones
    predictions = model.predict(h=horizon)
    predictions["MaterialName"] = material
    predictions["YearMonth"] = pd.date_range(start=datetime.now(), periods=horizon, freq="ME")
    predictions.set_index('YearMonth', inplace=True)

    return {
        "material": material,
        "predictions": predictions.to_dict(orient="records")
    }