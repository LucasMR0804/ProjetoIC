# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uIxwtzJjAQvPuaC4WH9EK3vAIImrnMKr
"""
import os
import streamlit as st
import openai
from openai import OpenAI

# Configurar a chave da API da OpenAI usando a variável de ambiente
openai.api_key = os.getenv('OPENAI_API_KEY')

# Função para obter a resposta da OpenAI GPT-3.5
client = OpenAI()
def generate_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    return stream["choices"][0]["message"]["content"].strip()

# Configurar a página do Streamlit
st.title("Chatbot com Streamlit e OpenAI")

# Adicionar um campo de texto para a pergunta do usuário
pergunta_usuario = st.text_input("Faça uma pergunta ao chatbot:")

# Verificar se a pergunta foi feita e obter a resposta
if pergunta_usuario:
    resposta_chatbot = generate_response(pergunta_usuario)
    st.write(resposta_chatbot)

# Adicionar uma seção de informações
st.info("Este é um chatbot simples usando Streamlit e OpenAI GPT-3.5.")
