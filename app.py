import streamlit as st
import os
from dotenv import load_dotenv
from html_template_1 import css, bot_template, user_template, logo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, SystemMessage

system_prompt = """
Eres Adri, un asistente de IA avanzado diseñado para la división GovLab de la Universidad de la Sabana. Tu objetivo principal es identificar oportunidades de consultoría basadas exclusivamente en los documentos proporcionados, generar ideas y estrategias de valor para implementarlas. 

Instrucciones:

Utiliza únicamente la información de los siguientes documentos para responder a la consulta:
<documentos>
{context}
</documentos>

2. El query al que debes responder es:
<query>
{query}
</query>

3. Analiza los documentos e identifica potenciales oportunidades de consultoría alineadas con las áreas de experiencia de la Universidad de La Sabana, las cuales son:
[
    "Hidrógeno y transición energética",
    "Gestión y gobierno TI",
    "Gestión y mejora de procesos",
    "Huella de carbono y emisiones",
    "Seguridad y salud en el trabajo",
    "Innovación",
    "Cargas laborales",
    "Estudios de vida útil",
    "Prototipado",
    "Valoración y diseño de productos",
    "Modelos de negocio",
    "Productividad",
    "Excelencia operativa",
    "Mejora continua, Lean Manufacturing y Kaizen",
    "Logística urbana, de salud y humanitaria",
    "Logística de abastecimiento, distribución y transporte",
    "Control de procesos",
    "Machine Learning",
    "Modelos de riesgo",
    "Factibilidad técnica, económica, ambiental",
    "Gestión estratégica",
    "Arquitectura de software",
    "Cultura ágil",
    "Ciencia de datos y analítica",
    "Liderazgo y power skills",
    "Servicio al cliente",
    "Competencias",
    "Gestión de proyectos",
    "Sistemas de información geográfica",
    "Sistemas de producción y gestión",
    "Simulación y robótica",
    "Sistemas de manufactura",
    "Inocuidad y seguridad alimentaria",
    "Caracterización y tratamiento de aguas",
    "Planificación territorial y urbana",
    "Movilidad urbana",
    "Biocompatibilidad de materiales",
    "Planeación y programación de producción",
    "Sistemas de gestión de calidad",
    "Inteligencia artificial",
    "Internet de las cosas",
    "Big Data",
    "Seguridad informática",
    "Control y automatización industrial"
]

4. Al responder:
- Asegúrate de seguir rigurosamente la información contenida en los documentos proporcionados. Al responder, menciona proyectos específicos relacionados con el para ofrecer ideas concretas y ejecutables, evitando generalidades. Proporciona múltiples propuestas que se puedan implementar.
- Si la información no está en los documentos, di "No tengo suficiente información para responder eso basado en los documentos proporcionados." No hagas suposiciones ni uses conocimientos externos.
- Si se te pregunta sobre una sección o página específica, proporciona esa información si está disponible.
- Si se te pregunta quién eres, responde "Soy Adri, tu IA para estar un paso adelante e identificar oportunidades de consultoría en entidades territoriales, pero todavia no respondas con oprtunidades de consultoria.s" 

5. Organiza tu respuesta de manera clara y concisa, proporcionando resúmenes ejecutivos cuando sea necesario.

6. Maintain a professional and constructive tone throughout the interaction.

Mantén un tono profesional y constructivo durante toda la interacción.

Ahora, por favor proporciona tu respuesta basándote únicamente en la información de los documentos y la consulta dada.
"""

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
PINE_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(PINE_KEY)
cloud = "aws"
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to create or get Pinecone index
def get_or_create_pinecone_index(index_name="oportunidades-consultoria"):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # dimension for text-embedding-3-small
                spec=spec,
                metric="cosine"  # Changed to cosine similarity
            )
            st.success(f"Created new Pinecone index: {index_name}")
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Error with Pinecone index: {str(e)}")
        return None

# Function to initialize vector store
def initialize_vector_store(index, documents):
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Split documents into chunks
        texts = []
        metadata = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            texts.extend(chunks)
            metadata.extend([{"source": "document", "text": chunk} for chunk in chunks])
        
        # Create vector store
        vector_store = PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name="oportunidades-consultoria"
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

def process_documents(uploaded_files):
    try:
        all_docs = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            file_name = os.path.join('./documents/', uploaded_file.name)
            
            with open(file_name, "wb") as f:
                f.write(bytes_data)
            
            # Load document based on file type
            name, extension = os.path.splitext(file_name)
            if extension.lower() in ['.pdf', '.PDF']:
                loader = PyPDFLoader(file_name)
            elif extension.lower() == '.docx':
                loader = Docx2txtLoader(file_name)
            elif extension.lower() == '.txt':
                loader = TextLoader(file_name)
            elif extension.lower() == '.csv':
                loader = CSVLoader(file_name)
            else:
                st.error(f'Unsupported format: {extension}')
                continue
                
            docs = loader.load()
            if docs:
                all_docs.extend(docs)
        
        if all_docs:
            index = get_or_create_pinecone_index()
            if index:
                vector_store = initialize_vector_store(index, all_docs)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success("Documents processed successfully!")
                    return True
        
        return False 
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

# StreamHandler for streaming responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(bot_template.replace("{{MSG}}", self.text), unsafe_allow_html=True)

def generate_response(query, model_choice):
    try:
        if not st.session_state.vector_store:
            return "PorFavor sube un documento primero."
        
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Use vector store for similarity search
        relevant_docs = st.session_state.vector_store.similarity_search(
            query,
            k=4,  # Number of relevant documents to retrieve
        )
        
        if not relevant_docs:
            return "No relevant information found in the documents."
        
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=st.session_state.get('temperature', 0.5),
            streaming=True,
            callbacks=[stream_handler]
        )
        
        formatted_system_prompt = system_prompt.format(context=context, query=query)
        messages = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=query)
        ]
        
        response = chat_model.invoke(messages)
        return stream_handler.text
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, an error occurred while processing your request."

# Streamlit UI
st.set_page_config(page_title="GOV", layout="centered")
st.markdown(logo, unsafe_allow_html=True)
st.title("ADRi-a")
st.markdown("**Buscador de oportunidades de consultoría en los gobiernos locales.**")

st.write(css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    model_choice = st.selectbox("Selecciona Modelo", ["OpenAI"])
    uploaded_files = st.file_uploader(
        "Sube tus Documentos ACÁ", 
        type=["pdf", "csv", "txt", "xlsx", "docx"], 
        accept_multiple_files=True
    )
    
    # Create documents directory if it doesn't exist
    documents_dir = './documents/'
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    
    # Process documents button
    if st.button("Cargar Documentos"):
        if uploaded_files:
            with st.spinner("Procesando documentos..."):
                success = process_documents(uploaded_files)
                if success:
                    st.success("¡Documentos procesados y listos para consulta!")
                else:
                    st.error("Error al procesar los documentos. Por favor, intente nuevamente.")
        else:
            st.warning("Por favor, sube algunos documentos primero.")
    
    # Chat history clear button
    if st.button("Borrar Historial Chat"):
        st.session_state.messages = []
        st.success("¡Historial de chat borrado!")
    
    # Temperature control
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    st.session_state.temperature = temperature
    st.caption("""Ajusta la creatividad 
    - Temperatura Baja: más preciso y consistente. 
    - Temperatura Alta: más creativo y variado.""")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Limit chat history
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]
    
    # Display user message
    with st.chat_message("U"):
        st.markdown(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)
    
    # Generate and display assistant response
    with st.chat_message("Adri"):
        full_response = generate_response(prompt, model_choice)
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response}) 
