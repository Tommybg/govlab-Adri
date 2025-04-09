# Frontend Module 
import streamlit as st 
#Python Modules 
import os 
import pandas as pd 
from dotenv import load_dotenv 
from html_template_1 import css, bot_template, user_template, logo
# Langchain modules
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever 
from langchain.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
#Pinecone modules
from pinecone import Pinecone, ServerlessSpec 
from pinecone_text.sparse import BM25Encoder
# Other Modules 
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()


#API KEYS CALLS 
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

PINE_KEY = os.environ.get("PINECONE_API_KEY") 
if PINE_KEY is None: 
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize Pinecone Client
pc = Pinecone(PINE_KEY)
cloud = "aws"
region = 'us-east-1' 
spec = ServerlessSpec(cloud=cloud, region=region)
bm25 = BM25Encoder().default()


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

# Create UPLOADED FILES folder 
documents_folder = './documents'
os.makedirs(documents_folder, exist_ok=True) 

# Define the Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content  # The text content of the document
        self.metadata = metadata if metadata is not None else {}  # Additional metadata

# Function to generate a unique CSV filename
def get_unique_filename(base_name, extension, folder): 
    counter = 0
    new_filename = f"{base_name}{extension}"
    
    while os.path.exists(os.path.join(folder, new_filename)):
        counter += 1
        new_filename = f"{base_name}_{counter}{extension}"
    
    return os.path.join(folder, new_filename)

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
    elif extension.lower() == ".xlsx":  
        excel_data = pd.read_excel(file, engine='openpyxl')
        base_name = os.path.splitext(os.path.basename(file))[0]
        file_csv = get_unique_filename(base_name, ".csv", documents_folder)
        excel_data.to_csv(file_csv, index=False)
        loader = CSVLoader(file_csv)
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

# Improved chunking function with contextual information
def chunk_data(docs, chunk_size=300, chunk_overlap=30):  # Reduced chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = []
    for doc in docs:
        doc_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(doc_chunks):
            # Limit context window to reduce overall size
            start = max(0, i - 1)
            end = min(len(doc_chunks), i + 1)  # Reduced context window
            context = "\n".join(doc_chunks[start:end])
            
            # Truncate context if it's too large
            if len(context) > 1000:  # Add size limit for context
                context = context[:1000] + "..."
            
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "context": context,
                    "chunk_id": i
                }
            ))
    return chunks

#Embedding creation using PINECONE 
def create_embeddings_pinecone(chunks, index_name="oportunidades-consultoria"):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small dimension
            spec=spec,
            metric="dotproduct"
        )
    
    index = pc.Index(index_name)
    
    # Initialize hybrid search retriever with optimal parameters
    vector_store = PineconeHybridSearchRetriever(
        index=index,
        embeddings=embeddings,
        sparse_encoder=bm25, 
        alpha = 0.70, 
        top_k = 4
    )

    try:
        # Process in optimized batches
        batch_size = 100  # Increased for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:min(i + batch_size, len(chunks))]
            
            # Prepare batch data
            vectors = []
            for j, doc in enumerate(batch):
                # Generate both dense and sparse embeddings
                dense_emb = embeddings.embed_documents([doc.page_content])[0]
                sparse_emb = bm25.encode_documents([doc.page_content])[0]
                
                vectors.append({
                    'id': f"{i}-{j}",
                    'values': dense_emb,
                    'sparse_values': sparse_emb,
                    'metadata': {
                        'source': doc.metadata.get('source', ''),
                        'chunk_id': doc.metadata.get('chunk_id')
                    }
                })
            
            index.upsert(vectors=vectors)
            
        return vector_store
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {e}")
        return None
    
# Update the load_vector_store function as well
def load_vector_store(index_name="oportunidades-consultoria"):
    # Check if index exists in Pinecone
    if index_name in pc.list_indexes().names():
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
        index = pc.Index(index_name)
        
        # Initialize retriever with existing index
        vector_store = PineconeHybridSearchRetriever(
            index=index,
            embeddings=embeddings,
            sparse_encoder=bm25,
            alpha = 0.70, 
            top_k = 4
        )
        return vector_store

# Update the update_embeddings_pinecone function
def update_embeddings_pinecone(chunks, vector_store, index_name="oportunidades-consultoria"):
    # Initialize vector store if None
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=API_KEY)
        index = pc.Index(index_name)
        vector_store = PineconeHybridSearchRetriever(
            index=index,
            embeddings=embeddings,
            sparse_encoder=bm25, 
            alpha = 0.70, 
            top_k = 4
            
        )
    
    try:
        # Get the index instance
        index = pc.Index(index_name)
        
        # Optimize batch processing
        batch_size = 100  # Increased for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:min(i + batch_size, len(chunks))]
            vectors = []
            
            for j, chunk in enumerate(batch):
                # Generate both dense and sparse embeddings
                dense_emb = vector_store.embed_documents([chunk.page_content])[0]
                sparse_emb = bm25.encode_documents([chunk.page_content])[0]
                
                # Create vector with both embeddings
                vectors.append({
                    'id': f"update-{i}-{j}",  # Unique ID for updated vectors
                    'values': dense_emb,
                    'sparse_values': sparse_emb,  # Add sparse embeddings
                    'metadata': {
                        'source': chunk.metadata.get('source', ''),
                        'chunk_id': chunk.metadata.get('chunk_id')
                        
                    }
                })
            
            # Batch upsert with progress indicator
            with st.spinner(f'Updating batch {i//batch_size + 1}...'):
                index.upsert(vectors=vectors)
        
        st.success(f"Successfully updated {len(chunks)} chunks in vector store")
        return vector_store
        
    except Exception as e:
        st.error(f"Error updating the Pinecone vector store: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
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

# Improved system prompt
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
- Si se te pregunta quién eres, responde "Soy Adri, tu IA para estar un paso adelante e identificar oportunidades de consultoría en entidades territoriales" 

5. Organiza tu respuesta de manera clara y concisa, proporcionando resúmenes ejecutivos cuando sea necesario.

6. Maintain a professional and constructive tone throughout the interaction.

Mantén un tono profesional y constructivo durante toda la interacción.

Ahora, por favor proporciona tu respuesta basándote únicamente en la información de los documentos y la consulta dada.
"""

# Update the generate_response function
def generate_response(q, model_choice):
    if st.session_state.vector_store is None:
        return "Por favor suba un documento primero."

    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)

        if model_choice == "OpenAI":
            chat_model = ChatOpenAI(
                model="gpt-4o",
                temperature=temperature_control,
                api_key=API_KEY,
                streaming=True,
                callbacks=[stream_handler]
            )
        elif model_choice == "Groq API":
            chat_model = None  # Placeholder for Groq API model
            st.warning("Groq API model is not yet implemented.")
        elif model_choice == "Claude":
            chat_model = None  # Placeholder for Claude model
            st.warning("Claude model is not yet implemented.")

        # Initialize base_retriever first
        retriever = st.session_state.vector_store

        relevant_docs = retriever.invoke(q)
        context = format_docs(relevant_docs)

        formatted_system_prompt = system_prompt.format(context=context, query=q)

        messages = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=q)
        ] + [HumanMessage(content=msg["content"]) for msg in st.session_state.messages]

        response = chat_model.invoke(messages)
        return stream_handler.text

    except Exception as e:
        st.error(f"Ocurrió un error al generar la respuesta: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Lo siento, pero ocurrió un error al procesar su solicitud. Por favor, intente de nuevo o contacte al soporte si el problema persiste."

# Streamlit UI 
st.set_page_config(page_title="GOV", layout="centered") 
st.markdown(logo, unsafe_allow_html=True)
st.title("ADRi-a", anchor=False)
st.markdown("**Buscador de oportunidades de consultoría en los gobiernos locales.**") 

st.write(css, unsafe_allow_html=True)

# Sidebar
with st.sidebar: 
    model_choice = st.selectbox("Selecciona el modelo deseado:", ["OpenAI", "Groq API", "Claude"])

    uploaded_files = st.file_uploader("Sube tus documentos ACÁ", type=["pdf", "csv", "txt", "xlsx", "docx"], accept_multiple_files=True)
    add_data = st.button("Cargar Documentos")
    
    if uploaded_files and add_data:
        all_chunks = []
        with st.spinner("Procesando sus Documentos..."):
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                file_name = os.path.join(documents_folder, uploaded_file.name)

                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                docs = load_document(file_name)
                if docs:
                    chunks = chunk_data(docs)
                    all_chunks.extend(chunks)

            # Check if we already have a vector store
            if st.session_state.vector_store is None:
                # Create new vector store if none exists
                vector_store = create_embeddings_pinecone(all_chunks)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success("¡Documentos procesados e índice creado con éxito!")
                else:
                    st.error("Error al crear el índice. Por favor intente de nuevo.")
            else:
                # Update existing vector store
                updated_store = update_embeddings_pinecone(
                    chunks=all_chunks,
                    vector_store=st.session_state.vector_store
                )
                if updated_store:
                    st.session_state.vector_store = updated_store
                    st.success("¡Documentos actualizados con éxito!")
                else:
                    st.error("Error al actualizar documentos. Por favor intente de nuevo.")

    if st.button("Borrar Historial Chat"):
        clear_chat_history()

    temperature_control = st.slider("Temperatura", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
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
