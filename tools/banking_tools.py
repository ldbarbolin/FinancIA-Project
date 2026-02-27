import json
import os
import pandas as pd
from langchain.tools import tool

# Definimos la ruta al archivo JSON para que lo encuentre sin importar desde dónde ejecutemos
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')

def cargar_bd():
    """Función auxiliar para leer el JSON simulando una llamada a base de datos."""
    try:
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@tool
def obtener_saldo(id_cliente: str) -> str:
    """
    Útil para consultar el saldo actual disponible en la cuenta bancaria del cliente.
    Devuelve el monto en Bolivianos (Bs.).
    Parámetro: id_cliente (string), por ejemplo "1001".
    """
    db = cargar_bd()
    cliente = db.get(id_cliente)
    
    if cliente:
        saldo = cliente["saldo_actual"]
        nombre = cliente["nombre"]
        return f"El saldo actual de la cuenta de {nombre} es de {saldo} Bs."
    return "Error: Cliente no encontrado en la base de datos."

@tool
def obtener_gastos_recientes(id_cliente: str) -> str:
    """
    Útil para obtener el historial de las últimas transacciones y gastos del cliente.
    Ideal para analizar en qué ha estado gastando su dinero.
    Parámetro: id_cliente (string), por ejemplo "1001".
    """
    db = cargar_bd()
    cliente = db.get(id_cliente)
    
    if cliente:
        gastos = cliente["gastos_recientes"]
        if not gastos:
            return "No hay transacciones recientes registradas para este cliente."
        
        # Formateamos el JSON a un texto claro para que el LLM lo procese más fácil
        resultado = f"Historial de gastos recientes para {cliente['nombre']}:\n"
        for gasto in gastos:
            resultado += f"- Fecha: {gasto['fecha']} | Monto: {gasto['monto']} Bs. | Detalle: {gasto['descripcion']}\n"
        return resultado
        
    return "Error: Cliente no encontrado en la base de datos."

@tool
def analizar_estadisticas_periodo(id_cliente: str, fecha_inicio: str = None, fecha_fin: str = None) -> str:
    """
    Útil para obtener un resumen estadístico de los gastos agrupados por categoría.
    El modelo puede usar esta herramienta para responder consultas sobre días, semanas, meses o años.
    Parámetros:
    - id_cliente: "1001"
    - fecha_inicio: Fecha inicial en formato "YYYY-MM-DD" (opcional).
    - fecha_fin: Fecha final en formato "YYYY-MM-DD" (opcional).
    Devuelve los totales gastados por categoría en el periodo solicitado.
    """
    ruta_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'gastos_historicos.csv')
    
    try:
        df = pd.read_csv(ruta_csv)
        df_cliente = df[df['id_cliente'] == int(id_cliente)]
        
        # Convertimos la columna de fechas a formato datetime de Pandas para hacer comparaciones exactas
        df_cliente['fecha'] = pd.to_datetime(df_cliente['fecha'])
        
        # Filtramos por fecha de inicio si el modelo la proporciona
        if fecha_inicio:
            inicio = pd.to_datetime(fecha_inicio)
            df_cliente = df_cliente[df_cliente['fecha'] >= inicio]
            
        # Filtramos por fecha de fin si el modelo la proporciona
        if fecha_fin:
            fin = pd.to_datetime(fecha_fin)
            df_cliente = df_cliente[df_cliente['fecha'] <= fin]
            
        if df_cliente.empty:
            return f"No se encontraron transacciones en el periodo solicitado."
            
        # Agrupamos por categoría y sumamos los montos
        resumen = df_cliente.groupby('categoria')['monto'].sum().reset_index()
        total_gastado = df_cliente['monto'].sum()
        
        # Formateamos el resultado
        periodo_texto = f"desde {fecha_inicio} hasta {fecha_fin}" if (fecha_inicio or fecha_fin) else "Histórico Total"
        resultado = f"--- Resumen de Gastos ({periodo_texto}) ---\n"
        resultado += f"Gasto Total: {total_gastado:.2f} Bs.\n"
        resultado += "Desglose por categoría:\n"
        for _, row in resumen.iterrows():
            resultado += f"- {row['categoria']}: {row['monto']:.2f} Bs.\n"
            
        return resultado
        
    except FileNotFoundError:
        return "Error: No se encontró la base de datos histórica."