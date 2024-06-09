# Bibliotecas e módulos usados
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv, find_dotenv
import os
import shutil
import streamlit as st
from langchain_openai import ChatOpenAI

# Busca da chave API no arquivo (.env)
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Diretórios utilizados [substituir DATA_PATH pelo local dos dados]
CHROMA_PATH = "chroma"
DATA_PATH = "C:\\Users\\Lukinha\\Documents\\Projeto IC"

# Gerador do espaço de armazenamento de dados
def generate_data_store():
    documents = load_documents()#Documentos gerados pela leitura da URL de referência
    chunks = split_text(documents)#Porções subdivididas dos documents gerados
    save_to_chroma(chunks)

# Carregador de dados
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    if not documents:
        st.error("Nenhum documento foi carregado. Verifique o caminho dos dados e o padrão do arquivo.")#Ponto de revisão de carregamento de documentos
    else:
        st.info(f"{len(documents)} documentos carregados com sucesso.")
    return documents

# Divisão dos dados a fins de simplificação
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(#Criterios de separação dos documentos em porções de tamanhos iguais
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        st.error("Nenhum chunk foi criado. Verifique o processo de divisão de texto.")#Ponto de verificação das porções subdivididas
    else:
        st.info(f"{len(chunks)} chunks criados com sucesso.")
    return chunks

# Criação do Espaço para o banco de dados
def save_to_chroma(chunks: list[Document]):
    # Se tiver a presença de um banco de dados anterior, esta porção irá realizar a 'limpeza' do espaço
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Criação do banco de dados que será usado para procura das 'querrys'
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_PATH
    )
    db.persist()
    st.info("Base de dados Chroma criada e persistida com sucesso.")#Confirmação da criação de dados

# Função principal
def main():
    # Preparando o banco de dados
    if not os.path.exists(CHROMA_PATH):
        generate_data_store()
    
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except ValueError as e:
        st.error(f"Erro ao conectar com o banco de dados Chroma: {str(e)}")#Ponto de validação da formação do banco de dados
        return
    
    # Título na interface
    st.title("CDVU-2024")
    
    # Mensagem inicial na interface
    st.info("Bem-vinde ao CDVU-2024, o seu Centro de Dúvidas sobre o Vestibular Unicamp 2024!")

    # Procura no banco de dados
    query_text = st.text_input("Digite a pergunta:") # Espaço de busca na interface
    if st.button("Buscar"):
        results = db.similarity_search_with_relevance_scores(query_text, k=3) # aplicação do RAG; busca dos 3 resultados mais próximos por distância euclidiana
        if len(results) == 0 or results[0][1] < 0.7: # Critério de aceitação mínima de resposta
            st.warning(f"O banco de dados é: {db}")
            st.warning(f'Os resultados foram: {results}')
            st.warning(f"Não foi possível encontrar resultados; pois {len(results)}")#Mensagem em caso de erro profundo
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) # Classificação das respostas por pontuação RAG
            prompt = f"Contexto:\n{context_text}\nPergunta:\n{query_text}"

            model = ChatOpenAI(api_key=openai_api_key)
            response_text = model.predict(prompt) # Modelo de treino para formulação da resposta pela IA

            sources = [doc.metadata.get("source", None) for doc, _score in results]
            formatted_response = f"Resposta: {response_text}" # Resposta formatada pela IA
            st.info(f'{formatted_response}') # Resposta na interface

if __name__ == "__main__":
    main()
    