#importacion de librerias previamente instal
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import io
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
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

#Panel principal con el chat
st.title("🤖 FinancIA")
st.markdown("Tu asesor financiero personal impulsado por Inteligencia Artificial Generativa.")

# Memoria del chat en la sesión
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "¡Hola David! Ya analicé tu dashboard de la izquierda. ¿Qué duda tienes sobre tus finanzas hoy?"}
    ]

# Mostrar historial de mensajes
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cuánto he gastado este mes en transporte?"):
    
    # 1. Mostrar pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    
    # 2. Inyectar silenciosamente el ID del cliente para que el agente sepa de quién buscar datos
    pregunta_enriquecida = f"Soy el cliente con ID 1001. {prompt}"
    
    # 3. Procesar respuesta del Agente (mostrando un 'spinner' de carga animado)
    with st.chat_message("assistant"):
        with st.spinner("Revisando tus números y calculando..."):
            try:
                respuesta = agent_executor.invoke({"messages": [HumanMessage(content=pregunta_enriquecida)]})
                respuesta_texto = respuesta["messages"][-1].content
                st.markdown(respuesta_texto)
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_texto})
                
                # --- NUEVO: Refrescar gráficos si se registró un gasto ---
                if "registrado" in respuesta_texto.lower() or "éxito" in respuesta_texto.lower():
                    cargar_datos_historicos.clear() # Limpia la caché de Pandas
                    st.rerun() # Recarga la app para que los gráficos se actualicen al instante
                    
            except Exception as e:
                st.error(f"Error de conexión: {e}")