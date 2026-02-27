#importacion de librerias previamente instal
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import io
import json
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent 
from tools.banking_tools import obtener_saldo, obtener_gastos_recientes, analizar_estadisticas_periodo, registrar_gasto


# Cargamos variables de entorno
load_dotenv('.env')

# Configuración de la página de Streamlit
st.set_page_config(page_title="FinancIA", page_icon="💰", layout="wide")

#Inicialización del modelo de lenguaje
@st.cache_resource
def iniciar_agente():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.2,
        max_retries=2
    )
    
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    instrucciones = f"""Eres FinancIA, un asesor financiero personal empático, analítico y directo.
    REGLA DE ORO TEMPORAL: Hoy es la fecha {fecha_hoy}. Usa esta fecha como tu punto de referencia.
    
    Tu objetivo es ayudar al usuario a entender sus finanzas y mejorar su capacidad de ahorro.
    Reglas:
    1. Usa las herramientas para consultar saldo o estadísticas.
    2. Si el usuario menciona que realizó un nuevo gasto, usa la herramienta 'registrar_gasto' y clasifica la categoría.
    3. Identifica "gastos hormiga" y da 2 o mas consejos accionables al final segun tu criterio.
    4. Comunícate en Bolivianos (Bs.) y usa un tono amigable pero profesional.
    5.REGLA DE FORMATO: Cuando el usuario te pida ver sus gastos, su historial o un resumen detallado, PRESÉNTALO SIEMPRE EN UNA TABLA ESTÉTICA USANDO FORMATO MARKDOWN (con columnas como Fecha, Descripción, Categoría y Monto).
    6. ADVERTENCIA: Nunca recomiendes inversiones específicas.
    """
    
    tools = [obtener_saldo, obtener_gastos_recientes, analizar_estadisticas_periodo, registrar_gasto]
    return create_react_agent(llm, tools, prompt=instrucciones)

agent_executor = iniciar_agente()

# Cargamos los datos históricos de gastos
@st.cache_data
def cargar_datos_historicos():
    ruta_csv = os.path.join(os.path.dirname(__file__), 'data', 'gastos_historicos.csv')
    try:
        df = pd.read_csv(ruta_csv)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        return pd.DataFrame()

df_gastos = cargar_datos_historicos()

#Panel lateral con gráficos de resumen
with st.sidebar:
    st.title("📊 Mi Dashboard Visual")
    st.write("Resumen automático de tu cuenta.")
    
    if not df_gastos.empty:
        # Simulamos que el usuario logueado es el 1001 (Carlos)
        df_cliente = df_gastos[df_gastos['id_cliente'] == 1001].copy()
        
        # Gráfico 1:(Gastos por Categoría)
        st.subheader("Distribución de Gastos")
        resumen_cat = df_cliente.groupby('categoria')['monto'].sum().reset_index()
        fig_pie = px.pie(resumen_cat, values='monto', names='categoria', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Gráfico 2:(Evolución por Mes)
        st.subheader("Evolución Histórica")
        df_cliente['mes'] = df_cliente['fecha'].dt.to_period('M').astype(str)
        resumen_mes = df_cliente.groupby('mes')['monto'].sum().reset_index()
        fig_bar = px.bar(resumen_mes, x='mes', y='monto', text_auto='.2f', 
                         labels={'mes': 'Mes', 'monto': 'Gasto (Bs)'},
                         color_discrete_sequence=['#2C3E50'])
        fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        #Botón de exportación de datos
        st.divider() # Línea separadora estética
        st.subheader("📥 Exportar Datos")
        st.write("Descarga tu historial completo para tus registros.")
        
        # Preparamos el archivo Excel en memoria
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Quitamos la columna id_cliente para que el Excel sea más limpio para el usuario
            df_exportar = df_cliente.drop(columns=['id_cliente'])
            df_exportar.to_excel(writer, index=False, sheet_name='Mis_Gastos_FinancIA')
        
        # Creamos el botón nativo de Streamlit
        st.download_button(
            label="Descargar Historial (Excel)",
            data=buffer,
            file_name=f"Historial_FinancIA_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.warning("No se encontraron datos en el historial.")

RUTA_HISTORIAL = os.path.join(os.path.dirname(__file__), 'data', 'historial_chat.json')

def cargar_memoria():
    """Carga el historial del chat desde un archivo JSON."""
    if os.path.exists(RUTA_HISTORIAL):
        with open(RUTA_HISTORIAL, 'r', encoding='utf-8') as f:
            return json.load(f)
    return [
        {"role": "assistant", "content": "¡Hola Carlos! Ya analicé tu dashboard. ¿Qué duda tienes sobre tus finanzas hoy?"}
    ]

def guardar_memoria(mensajes):
    """Guarda el historial actual en un archivo JSON."""
    with open(RUTA_HISTORIAL, 'w', encoding='utf-8') as f:
        json.dump(mensajes, f, ensure_ascii=False, indent=4)


#Panel principal con el chat
st.title("🤖 FinancIA")
st.markdown("Tu asesor financiero personal impulsado por Inteligencia Artificial Generativa.")

# Inicializar o cargar la memoria desde el JSON
if "mensajes" not in st.session_state:
    st.session_state.mensajes = cargar_memoria()

# Mostrar historial de mensajes en la pantalla
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cuánto he gastado este mes en transporte?"):
    
    # 1. Mostrar pregunta del usuario y guardarla
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    guardar_memoria(st.session_state.mensajes) # Guardamos en el JSON al instante
    
    # 2. Construir la memoria conversacional para el Agente
    # Convertimos nuestro JSON a objetos HumanMessage y AIMessage que LangChain entiende
    historial_langchain = []
    for msg in st.session_state.mensajes:
        if msg["role"] == "user":
            # Inyectamos silenciosamente el ID solo en el último mensaje
            if msg == st.session_state.mensajes[-1]:
                historial_langchain.append(HumanMessage(content=f"Soy el cliente con ID 1001. {msg['content']}"))
            else:
                historial_langchain.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            historial_langchain.append(AIMessage(content=msg["content"]))
    
    # 3. Procesar respuesta del Agente
    with st.chat_message("assistant"):
        with st.spinner("Revisando tus números y calculando..."):
            try:
                # Le pasamos TODO el historial, no solo la última pregunta
                respuesta = agent_executor.invoke({"messages": historial_langchain})
                respuesta_texto = respuesta["messages"][-1].content
                st.markdown(respuesta_texto)
                
                # Guardamos la respuesta de la IA
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_texto})
                guardar_memoria(st.session_state.mensajes) # Guardamos en el JSON al instante
                
                # Refrescar gráficos si se registró un gasto
                if "registrado" in respuesta_texto.lower() or "éxito" in respuesta_texto.lower():
                    cargar_datos_historicos.clear()
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error de conexión: {e}")