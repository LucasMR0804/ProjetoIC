# Projeto IC
# Passos para utilização dessa aplicação:

Esta é uma aplicação de exemplo utilizando Streamlit, LangChain e OpenAI para busca e análise de documentos.

## Pré-requisitos

- Python 3.8+
- Conta na OpenAI com uma chave API válida

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu_usuario/seu_repositorio.git
    cd seu_repositorio
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure as variáveis de ambiente:
    - Copie o arquivo `.env.example` para `.env` e preencha suas credenciais.
    ```bash
    cp .env.example .env
    ```

    - Edite o arquivo `.env` para adicionar sua chave da API OpenAI:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Execução

Para executar a aplicação, use o Streamlit:
```bash
streamlit run project_final.py
