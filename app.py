import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from datasets import load_dataset

# Baixa o dataset e salva o arquivo JSONL localmente
dataset = load_dataset("gretelai/symptom_to_diagnosis", split="train")
dataset.to_json("data/train.jsonl")

# 1. Carrega o mapeamento de sintomas
def carregar_dados():
    with open("data/sintomas_especialidades.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 2. Processa o texto do usuário com NLP
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

def extrair_palavras_chave(texto):
    doc = nlp(texto.lower())
    palavras_chave = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    return " ".join(palavras_chave)

# 3. Encontra a especialidade mais relevante
def encontrar_especialidade(texto, sintomas_especialidades):
    palavras_chave = extrair_palavras_chave(texto)
    sintomas = list(sintomas_especialidades.keys())
    
    # Vetorização TF-IDF
    vetorizador = TfidfVectorizer()
    vetores_sintomas = vetorizador.fit_transform(sintomas)
    vetor_usuario = vetorizador.transform([palavras_chave])
    
    # Comparação de similaridade
    similaridades = cosine_similarity(vetor_usuario, vetores_sintomas).flatten()
    indice_max = similaridades.argmax()
    
    return sintomas_especialidades[sintomas[indice_max]] if similaridades[indice_max] > 0.3 else "Clínico Geral"

# 4. Interface do usuário
def main():
    sintomas_especialidades = carregar_dados()
    
    print("Descreva seus sintomas (ex.: 'dor no peito e falta de ar'):")
    texto_usuario = input("> ")
    
    especialidade = encontrar_especialidade(texto_usuario, sintomas_especialidades)
    print(f"\nRecomendação: Consulte um {especialidade}.")

if __name__ == "__main__":
    main()