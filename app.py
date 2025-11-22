"""
🤖 CHATBOT DE EFICIENCIA ENERGÉTICA DE EDIFICIOS
Asistente conversacional para identificar edificios que requieren inspección.
Bootcamp G324 IA - Talento Tech - Armenia Quindío 2025
"""
import gradio as gr
import pandas as pd
import joblib
import os
import sklearn

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DEL MODELO
# ==============================================================================
PIPELINE_PATH = "pipeline_proyecto_emisiones.pkl"
MODELO_CARGADO = False
pipeline = None

try:
    if os.path.exists(PIPELINE_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        
        # --- PARCHE DE SEGURIDAD ---
        # Forzamos al encoder a ignorar categorías desconocidas (ej: un número de m2 nuevo)
        try:
            preprocessor = pipeline.named_steps['preprocesamiento']
            ohe = preprocessor.named_transformers_['cat']
            ohe.handle_unknown = 'ignore'
            print("🔧 Parche aplicado: handle_unknown='ignore'")
        except Exception as e:
            print(f"⚠️ Nota: No se requirió parche o falló: {e}")
        # ---------------------------
        
        MODELO_CARGADO = True
        print("✅ Pipeline cargado correctamente")
    else:
        print("⚠️ Archivo de pipeline no encontrado")
except Exception as e:
    print(f"❌ Error crítico: {e}")

# ==============================================================================
# 2. LISTAS Y MAPEOS
# ==============================================================================
# Definimos las opciones válidas para validar las respuestas del usuario
OPCIONES_TIPO = ["Vivienda individual", "Bloque completo", "Local", "Unifamiliar", "Edificio completo"]
OPCIONES_PROVINCIA = ["ZARAGOZA", "HUESCA", "TERUEL"]
OPCIONES_CLASIFICACION = ["A", "B", "C", "D", "E", "F", "G"]

# Mapeos para codificar la entrada al modelo (Deben coincidir con tu entrenamiento)
TIPO_EDIFICIO_MAP = {k: i for i, k in enumerate(OPCIONES_TIPO)}
PROVINCIA_MAP = {k: i for i, k in enumerate(OPCIONES_PROVINCIA)}

# ==============================================================================
# 3. LÓGICA DEL CHAT (Preguntas paso a paso)
# ==============================================================================

# Lista de pasos: El bot irá recorriendo esta lista
PREGUNTAS = [
    {
        "clave": "inicio",
        "texto": "¡Hola! 👋 Soy tu Asistente de Eficiencia Energética.\n\nVoy a hacerte unas preguntas breves para evaluar si tu edificio necesita inspección urgente.\n\n👉 Para empezar, escribe: **'hola'**."
    },
    {
        "clave": "tipo_edificio",
        "texto": f"1️⃣ ¿Qué **tipo de edificio** es?\n\nOpciones válidas:\n- {', '.join(OPCIONES_TIPO)}",
        "opciones": OPCIONES_TIPO
    },
    {
        "clave": "superficie_m2",
        "texto": "2️⃣ ¿Cuál es la **superficie** aproximada en metros cuadrados (m²)?\n(Escribe solo el número, ej: 120)"
    },
    {
        "clave": "anio_construccion",
        "texto": "3️⃣ ¿En qué **año** se construyó el edificio?\n(Ej: 1990)"
    },
    {
        "clave": "provincia",
        "texto": f"4️⃣ ¿En qué **provincia** se encuentra?\n\nOpciones: {', '.join(OPCIONES_PROVINCIA)}",
        "opciones": OPCIONES_PROVINCIA
    },
    {
        "clave": "clasificacion_consumo",
        "texto": "5️⃣ ¿Cuál es su **Clasificación de Consumo** actual?\n(Opciones: A, B, C, D, E, F, G)",
        "opciones": OPCIONES_CLASIFICACION
    },
    {
        "clave": "consumo_kwh",
        "texto": "6️⃣ ¿Cuál es el **Consumo** en kWh/m²/año?\n(Ej: 150.5)"
    },
    {
        "clave": "emision_co2",
        "texto": "7️⃣ ¿Cuál es la **Emisión de CO₂** en kg/m²/año?\n(Ej: 35)"
    },
    {
        "clave": "anio_emision",
        "texto": "8️⃣ Por último, ¿En qué **año** se emitió el certificado energético?\n(Ej: 2020)"
    }
]

def validar_respuesta(texto, paso_actual):
    """Verifica si lo que escribió el usuario es válido para la pregunta actual"""
    pregunta = PREGUNTAS[paso_actual]
    clave = pregunta["clave"]
    texto = str(texto).strip()

    # Si es el saludo inicial, aceptamos cualquier cosa
    if clave == "inicio":
        return True, "ok"
    
    # Si la pregunta tiene opciones cerradas (Dropdown)
    if "opciones" in pregunta:
        opciones_lower = [o.lower() for o in pregunta["opciones"]]
        if texto.lower() in opciones_lower:
            # Devolvemos el texto con las mayúsculas correctas
            indice = opciones_lower.index(texto.lower())
            return True, pregunta["opciones"][indice]
        else:
            return False, f"⚠️ Opción no reconocida. Por favor elige una de: {', '.join(pregunta['opciones'])}"

    # Si la pregunta espera un número (superficie, año, consumo...)
    if clave in ["superficie_m2", "anio_construccion", "consumo_kwh", "emision_co2", "anio_emision"]:
        try:
            valor = float(texto)
            if valor < 0: return False, "⚠️ El número no puede ser negativo."
            if "anio" in clave and (valor < 1800 or valor > 2100): return False, "⚠️ El año no parece válido."
            return True, valor
        except ValueError:
            return False, "⚠️ Por favor ingresa un número válido (usa punto '.' para decimales)."

    return True, texto

def generar_prediccion(datos):
    """Toma los datos recolectados y consulta al modelo ML"""
    if not MODELO_CARGADO:
        return "❌ Error: El modelo no está cargado."

    try:
        # 1. Prepara los datos igual que en el entrenamiento
        # Importante: Convertir a int -> str para superficie y emision si se entrenaron así
        sup_str = str(int(datos["superficie_m2"]))
        emi_str = str(int(datos["emision_co2"]))
        
        tipo_cod = TIPO_EDIFICIO_MAP.get(datos["tipo_edificio"], 0)
        provincia_cod = PROVINCIA_MAP.get(datos["provincia"].upper(), 0)
        
        # Mapeo manual de clasificación (A-E=0, F-G=1)
        clasif_letra = datos["clasificacion_consumo"]
        clasif_cod = 1 if clasif_letra in ['F', 'G'] else 0

        # DataFrame de entrada
        entrada = pd.DataFrame({
            'clasificacion_consumo': [clasif_cod],
            'consumokwhm2anio': [datos["consumo_kwh"]],
            'tipo_edificio': [tipo_cod],
            'provincia': [provincia_cod],
            'anio_emision': [datos["anio_emision"]],
            'anio_construccion': [datos["anio_construccion"]],
            'superficie_m2': [sup_str],
            'emision_co2': [emi_str]
        })

        # 2. Predicción
        prediccion = pipeline.predict(entrada)[0]
        
        # 3. Probabilidades (si el modelo lo soporta)
        try:
            probs = pipeline.predict_proba(entrada)[0]
            prob_inef = probs[1] * 100
        except:
            prob_inef = 0

        # 4. Mensaje final
        antiguedad = 2025 - datos["anio_construccion"]
        
        if prediccion == 1:
            return (
                f"### 🔴 RESULTADO: INEFICIENTE\n\n"
                f"⚠️ **Este edificio requiere inspección urgente.**\n"
                f"La probabilidad de ineficiencia es del **{prob_inef:.1f}%**.\n\n"
                f"**Resumen del análisis:**\n"
                f"- Antigüedad: {antiguedad} años\n"
                f"- Consumo: {datos['consumo_kwh']} kWh/m²\n"
                f"- Emisiones: {datos['emision_co2']} kgCO₂/m²\n\n"
                f"💡 **Recomendación:** Contactar a un auditor energético para evaluar reformas de aislamiento."
            )
        else:
            return (
                f"### 🟢 RESULTADO: EFICIENTE\n\n"
                f"✅ **El edificio se encuentra en buen estado.**\n"
                f"No se detecta necesidad de intervención inmediata.\n\n"
                f"**Resumen del análisis:**\n"
                f"- Antigüedad: {antiguedad} años\n"
                f"- Consumo: {datos['consumo_kwh']} kWh/m²\n\n"
                f"💡 **Recomendación:** Mantener revisiones periódicas cada 5 años."
            )

    except Exception as e:
        return f"❌ Ocurrió un error interno al calcular: {str(e)}"

def responder(mensaje, historia, estado_actual):
    """
    Función principal del Chatbot.
    Maneja el flujo de conversación:
    1. Revisa en qué paso estamos.
    2. Valida la respuesta del usuario.
    3. Pasa al siguiente paso o da el resultado.
    """
    # estado_actual es una lista: [paso (int), datos (dict)]
    if estado_actual is None:
        estado_actual = [0, {}]
    
    paso, datos = estado_actual
    
    # Si es el primer mensaje (o el usuario dice reiniciar)
    if paso == 0:
        bot_msg = PREGUNTAS[1]["texto"]
        return bot_msg, [1, {}]

    # Validar respuesta del paso ANTERIOR
    es_valido, valor_validado = validar_respuesta(mensaje, paso)
    
    if not es_valido:
        # Si falla, repetimos la pregunta o damos error, pero no avanzamos paso
        return f"{valor_validado}\n\nIntenta de nuevo.", [paso, datos]
    
    # Guardar dato validado
    clave_actual = PREGUNTAS[paso]["clave"]
    datos[clave_actual] = valor_validado
    
    # Avanzar al siguiente paso
    nuevo_paso = paso + 1
    
    # Si ya terminamos todas las preguntas
    if nuevo_paso >= len(PREGUNTAS):
        mensaje_final = generar_prediccion(datos)
        mensaje_final += "\n\n🔄 **Escribe 'empezar' si quieres analizar otro edificio.**"
        return mensaje_final, [0, {}] # Reiniciamos estado para la próxima
    
    # Si faltan preguntas, enviamos la siguiente
    siguiente_pregunta = PREGUNTAS[nuevo_paso]["texto"]
    return siguiente_pregunta, [nuevo_paso, datos]

# ==============================================================================
# 4. INTERFAZ GRÁFICA (ChatInterface)
# ==============================================================================

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(theme=theme, title="Chatbot Energético") as demo:
    gr.Markdown("# 🤖 Chatbot de Auditoría Energética")
    gr.Markdown("Conversa con el asistente para diagnosticar tu edificio paso a paso.")
    
    # CORRECCIÓN AQUÍ: Inicializar con formato de lista de diccionarios
    chatbot = gr.Chatbot(
        label="Conversación",
        value=[{"role": "assistant", "content": PREGUNTAS[0]["texto"]}], 
        height=500,
        type="messages" # Nuevo formato de Gradio
    )
    
    msg = gr.Textbox(label="Tu respuesta", placeholder="Escribe aquí y presiona Enter...")
    clear = gr.Button("Reiniciar Chat")
    
    # Estado: guarda [numero_pregunta, diccionario_datos]
    estado = gr.State([0, {}]) 

    def user_turn(user_message, history, state):
        # Añade mensaje del usuario al chat
        # history es una lista de diccionarios [{'role': 'user', 'content': 'hola'}]
        return "", history + [{"role": "user", "content": user_message}], state

    def bot_turn(history, state):
        # history[-1] es el último mensaje (del usuario)
        user_message = history[-1]["content"]
        
        bot_message, new_state = responder(user_message, history, state)
        
        history.append({"role": "assistant", "content": bot_message})
        return history, new_state

    # Flujo de eventos
    msg.submit(user_turn, [msg, chatbot, estado], [msg, chatbot, estado]).then(
        bot_turn, [chatbot, estado], [chatbot, estado]
    )
    
    # CORRECCIÓN AQUÍ: La función reiniciar debe devolver una lista de diccionarios, no una lista de listas
    def reiniciar():
        return [{"role": "assistant", "content": PREGUNTAS[0]["texto"]}], [0, {}]
        
    clear.click(reiniciar, None, [chatbot, estado])

if __name__ == "__main__":
    demo.launch()