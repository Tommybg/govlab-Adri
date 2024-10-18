import streamlit as st 
import os 
import pandas as pd 
from dotenv import load_dotenv 
from html_template_1 import css, bot_template, user_template,logo
from langchain_openai  import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize vector_store in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []
    st.success("Historial de chat y memoria borrados!")

# Function to load document
@st.cache_data
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension.lower() in ['.pdf', '.PDF']:
        loader = PyPDFLoader(file)
    elif extension.lower() == '.docx':
        loader = Docx2txtLoader(file)
    elif extension.lower() == '.txt':
        loader = TextLoader(file)
    elif extension.lower() == '.csv':
        loader = CSVLoader(file) 
    else:
        st.error('Formato de documento no soportado!')
        return None

    try:
        docs = loader.load()
        if docs is None or len(docs) == 0:
            st.error("No se pudieron cargar documentos desde el archivo.")
            return None
        return docs
    except Exception as e:
        st.error(f"Error al cargar el documento: {str(e)}")
        return None

# Function to chunk data
def chunk_data(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Function to create embeddings and Chroma DB 
def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
    try:
        # Initialize Chroma with persistence
        vector_store = Chroma.from_documents(chunks, 
                                             embeddings, 
                                             persist_directory=None, 
                                             in_memory= True
                                             )
        return vector_store
    except Exception as e:
        st.error(f"Error al crear el vector store de Chroma: {e}")
        return None

# Load existing vector store if it exists
def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
        vector_store = Chroma(persist_directory=None,
                               embedding_function=embeddings, 
                               in_memory= True)
        return vector_store
    except Exception as e:
        st.error(f"Error al cargar el vector store de Chroma: {e}")
        return None

def update_embeddings_chroma(chunks, vector_store):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
    if vector_store:
        vector_store.add_documents(chunks)  # Append new documents
    else:
        vector_store = Chroma.from_documents(chunks, 
                                             embeddings, 
                                             persist_directory=None, 
                                             in_memory= True)
    vector_store.persist()
    return vector_store


# Load the vector store at the start
st.session_state.vector_store = load_vector_store()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Update the StreamHandler class
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(bot_template.replace("{{MSG}}", self.text), unsafe_allow_html=True)

# Function to truncate messages
def truncate_messages(max_messages=10):
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]

# Update the generate_response function
def generate_response(q, model_choice):
    if st.session_state.vector_store is None:
        return "Por favor suba un documento primero."

    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)

        # Select the model based on user choice
        if model_choice == "OpenAI":
            chat_model = ChatOpenAI(
                model="gpt-4o",  # Update this to the correct model name
                temperature= temperature_control,
                api_key=API_KEY,
                streaming=True,
                callbacks=[stream_handler]
            )
    
        elif model_choice == "Groq API":
            # Replace with Groq API model initialization
            chat_model = None  # Placeholder for Groq API model
            st.warning("Groq API model is not yet implemented.")
        elif model_choice == "Claude":
            # Replace with Claude model initialization
            chat_model = None  # Placeholder for Claude model
            st.warning("Claude model is not yet implemented.")

        system_prompt =  """
    Eres Adri, un asistente de IA avanzado diseñado para la división del GovLab de la Universidad de la Sabana. Tu objetivo principal es identificar y crear oportunidades de consultoría, basándote exclusivamente en los planes de desarrollo proporcionados.

    ## Contexto y Misión

    El GovLab es un laboratorio de innovación de la Universidad de La Sabana dedicado a buscar soluciones a problemas públicos mediante técnicas innovadoras, analítica de datos, co-creación y colaboración intersectorial.

    ## Tus Capacidades y Responsabilidades

    1. **Análisis Basado en Documentos:**
        - Usa técnicas de RAG (Retrieval-Augmented Generation) para extraer información relevante de los documentos y planes de desarrollo municipales y departamentales proporcionados.
        - Responde únicamente con hechos basados en los documentos cargados. No hagas suposiciones ni respondas con información fuera de los documentos.

    2. **Identificación de Oportunidades de Consultoría:**
        - Detecta áreas clave de interés o necesidad en los planes de desarrollo donde el GovLab pueda ofrecer consultoría valiosa.
        - Las oportunidades de consultoría deben estar alineadas con las siguientes áreas de especialidad de la Universidad de La Sabana:
        
        ### Áreas de Consultoría de la Universidad de La Sabana:
        - Hidrógeno y transición energética
        - Gestión y gobierno TI
        - Gestión y mejora de procesos
        - Huella de carbono y emisiones
        - Seguridad y salud en el trabajo
        - Innovación
        - Cargas laborales
        - Estudios de vida útil
        - Prototipado
        - Valoración y diseño de productos
        - Modelos de negocio
        - Productividad
        - Excelencia operativa
        - Mejora continua, Lean Manufacturing y Kaizen
        - Logística urbana, de salud y humanitaria
        - Logística de abastecimiento, distribución y transporte
        - Control de procesos
        - Machine Learning
        - Modelos de riesgo
        - Factibilidad técnica, económica, ambiental
        - Gestión estratégica
        - Arquitectura de software
        - Cultura ágil
        - Ciencia de datos y analítica
        - Liderazgo y power skills
        - Servicio al cliente
        - Competencias
        - Gestión de proyectos
        - Sistemas de información geográfica
        - Sistemas de producción y gestión
        - Simulación y robótica
        - Sistemas de manufactura
        - Inocuidad y seguridad alimentaria
        - Caracterización y tratamiento de aguas
        - Planificación territorial y urbana
        - Movilidad urbana
        - Biocompatibilidad de materiales
        - Planeación y programación de producción
        - Sistemas de gestión de calidad
        - Inteligencia artificial
        - Internet de las cosas
        - Big Data
        - Seguridad informática
        - Control y automatización industrial

    3. **Generación de Propuestas y Valor:**
        - Propón estrategias o ideas innovadoras solo cuando estén sustentadas en los hechos del documento.
        - Si la pregunta requiere más información que no está contenida en el documento o en tu base de conocimientos, indica que no tienes suficiente información, o pide más contexto al usuario para continuar.

    4. **Alineación con el Contexto Gubernamental:**
        - Verifica que las oportunidades propuestas estén alineadas con los objetivos del gobierno colombiano y con las particularidades de la región de interés.

    ## Directrices de Interacción y Límites

    - **Precisión:** Responde solo con la información basada en el texto del documento. Si no puedes responder de manera precisa basándote en los documentos o en tu base de conocimiento, pide más contexto o informa que no tienes suficiente información en ese momento.
    - **Identificación de Sección o Página:** Si el usuario te pregunta de qué parte, sección o página del documento obtuviste una información, responde indicando la sección, título o número de página en la medida de lo posible. Si no es posible localizar una parte exacta, proporciona el contexto general o pide más detalles.
    - **Saludos y Preguntas Generales:** Si el usuario te saluda o pregunta quién eres, responde de forma amable sin devolver una respuesta vacía, sin indicar que no tienes información y tu objetivo principal.
    - **Solicita Claridad:** Si una pregunta es ambigua o falta información en los documentos proporcionados, solicita aclaraciones o un contexto más amplio antes de generar una respuesta.
    - **Profesionalismo:** Mantén un tono profesional y constructivo en todas las interacciones.
    - **Presentación:** Organiza las respuestas de manera clara, concisa y estructurada, proporcionando resúmenes ejecutivos cuando sea necesario, pero basados siempre en los documentos cargados.

    ## Instrucciones Internas (para uso de IA, no visibles para el usuario final):

    1. **DOCUMENT:** (texto del documento utilizado para responder la pregunta)
    2. **QUESTION:** (pregunta del usuario)
    3. **ANSWER:** Responde a la pregunta del usuario utilizando solo los datos del DOCUMENT proporcionado. Si no tienes suficiente información para responder, solicita más contexto o explica que no tienes los datos necesarios.
    4. Si la pregunta es un saludo o una pregunta sobre quién eres, responde de manera corta y menciona tu objetivo principal resumido.
    5. Si el usuario pregunta de qué sección o página específica del documento proviene la respuesta, busca esa información en el DOCUMENT proporcionado y ofrécele el título de la sección o el número de página. Si no puedes encontrar la sección exacta, solicita más contexto o proporciona una referencia aproximada al documento.

    Nota: No incluyas el encabezado DOCUMENT, QUESTION o INSTRUCTIONS en la respuesta final. El usuario solo debe ver la respuesta sin ningún formato adicional.
"""





        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(q)  # Use invoke instead of get_relevant_documents
        context = format_docs(relevant_docs)

        # Include previous messages in the context
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}\n\nQuestion: {q}\nAnswer:")
        ] + [HumanMessage(content=msg["content"]) for msg in st.session_state.messages]

        response = chat_model.invoke(messages)  # Use invoke instead of __call__
        return stream_handler.text  # Return the full streamed text

    except Exception as e:
        st.error(f"Ocurrió un error al generar la respuesta: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Lo siento, pero ocurrió un error al procesar su solicitud. Por favor, intente de nuevo o contacte al soporte si el problema persiste."

# Streamlit UI 
st.set_page_config(page_title="GOV", layout="centered") 
st.markdown(logo, unsafe_allow_html=True)
st.title("ADRi-a", anchor= False)
st.markdown("**Buscador de oportunidades de consultoría en los gobiernos locales.**") 

st.write(css, unsafe_allow_html=True)

# Sidebar
with st.sidebar: 
    # Dropdown menu for model selection
    model_choice = st.selectbox("Selecciona el modelo deseado:", ["OpenAI", "Groq API", "Claude"])

    # Container para subir los documentos 
    uploaded_files = st.file_uploader("Sube tus documentos ACÁ", type=["pdf", "csv", "txt", "xlsx", "docx"], accept_multiple_files=True)
    # Boton para procesar los documentos y subirlos a la chroma db
    add_data = st.button("Cargar Documentos")
    
    if uploaded_files and add_data:
        all_chunks = []  # To store all chunks from multiple documents
        with st.spinner("Procesando sus Documentos..."):
            for i, uploaded_file in enumerate(uploaded_files):
                bytes_data = uploaded_file.read()

                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                docs = load_document(file_name)
                if docs:
                    chunks = chunk_data(docs)
                    all_chunks.extend(chunks)

            vector_store = create_embeddings_chroma(all_chunks)

            if vector_store:
                st.session_state.vector_store = vector_store
                st.success("Documento(s) procesado(s) con éxito!")
            else:
                st.error("Error en procesar documentos. Por favor intente de nuevo.") 

    if st.button("Borrar Historial Chat"):
        clear_chat_history()

    # Slider para el control de la temperatura por el Usuario 
    temperature_control = st.slider("Temperatura", min_value= 0.1, max_value= 1.0, value=0.5, step= 0.1)
    st.caption("""Ajusta la creatividad de las respuestas 
- Baja Temperatura: más precisas y coherentes. 
- Alta Temperatura: más creativas y variadas.""")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

# Chat Input 
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Truncate messages to keep the last N messages
    truncate_messages(max_messages=10)

    with st.chat_message("U"):
        st.markdown(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)

    with st.chat_message("Adri"):
        full_response = generate_response(prompt, model_choice)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
