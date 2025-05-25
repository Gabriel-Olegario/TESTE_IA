import spacy  # Biblioteca para processamento de linguagem natural (NLP)
import json  # Para manipulação de arquivos JSON
from sklearn.feature_extraction.text import TfidfVectorizer  # Para vetorização de texto (não usado diretamente aqui)
from sklearn.metrics.pairwise import cosine_similarity  # Para calcular similaridade de cosseno (não usado diretamente aqui)
from sentence_transformers import SentenceTransformer, util  # Para embeddings de sentenças e cálculo de similaridade

# Carrega o modelo BERT multilíngue para gerar embeddings de sentenças
bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def carregar_dados_train():
    """
    Lê o arquivo train.jsonl e carrega os dados de sintomas e diagnósticos.
    Cada linha do arquivo deve ser um JSON com chaves como 'input_text' e 'output_text'.
    Retorna uma lista de dicionários.
    """
    dados_sintomas = []
    with open("data/train.jsonl", "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    dados_sintomas.append(json.loads(linha))  # Adiciona o JSON decodificado à lista
                except json.JSONDecodeError:
                    continue  # Ignora linhas com erro de decodificação
    return dados_sintomas

def carregar_dados_especialista():
    """
    Lê o arquivo especialista.jsonl e carrega o mapeamento de diagnóstico para especialista.
    Suporta dois formatos de chave: 'input_text'/'output_text' ou 'doenca'/'especialista'.
    Retorna um dicionário: {diagnóstico: especialista}
    """
    especialistas = {}
    with open("data/especialista.jsonl", "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    registro = json.loads(linha)
                    # Tenta as duas possibilidades de chave
                    if "input_text" in registro and "output_text" in registro:
                        especialistas[registro["input_text"].lower()] = registro["output_text"]
                    elif "doenca" in registro and "especialista" in registro:
                        especialistas[registro["doenca"].lower()] = registro["especialista"]
                except json.JSONDecodeError:
                    continue  # Ignora linhas com erro de decodificação
    return especialistas

# Tenta carregar o modelo de linguagem do spaCy para português.
# Se não estiver instalado, faz o download automaticamente.
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

def extrair_palavras_chave(texto):
    """
    Extrai palavras-chave (substantivos, adjetivos e verbos) do texto usando spaCy.
    Retorna uma string com as palavras lematizadas separadas por espaço.
    """
    doc = nlp(texto.lower())
    palavras_chave = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    return " ".join(palavras_chave)

def encontrar_diagnostico_bert(texto, train_data):
    """
    Recebe o texto do usuário e os dados de treinamento.
    Calcula o embedding do texto do usuário e dos sintomas do banco usando BERT.
    Retorna o diagnóstico mais similar se a similaridade for maior que o threshold (0.5).
    Caso contrário, retorna 'Diagnóstico indefinido'.
    """
    sintomas = [item["input_text"] for item in train_data]  # Lista de sintomas do banco
    saidas = [item["output_text"] for item in train_data]   # Lista de diagnósticos correspondentes

    # Embedding do texto do usuário
    emb_usuario = bert_model.encode([texto], convert_to_tensor=True)
    # Embeddings dos sintomas do banco
    emb_sintomas = bert_model.encode(sintomas, convert_to_tensor=True)

    # Calcula similaridade de cosseno entre o texto do usuário e todos os sintomas do banco
    similaridades = util.pytorch_cos_sim(emb_usuario, emb_sintomas)[0]
    indice_max = int(similaridades.argmax())  # Índice do sintoma mais similar
    score_max = float(similaridades[indice_max])  # Valor da similaridade máxima

    # Retorna o diagnóstico se a similaridade for suficiente, senão retorna indefinido
    return saidas[indice_max] if score_max > 0.5 else "Diagnóstico indefinido"

def main():
    """
    Função principal do programa.
    - Carrega os dados de treinamento e especialistas.
    - Solicita ao usuário a descrição dos sintomas.
    - Encontra o diagnóstico mais provável usando BERT.
    - Busca o especialista recomendado para o diagnóstico.
    - Exibe o resultado ao usuário.
    """
    train = carregar_dados_train()
    especialistas = carregar_dados_especialista()

    print("Descreva seus sintomas (ex.: 'dor no peito e falta de ar'):")
    texto_usuario = input("> ")

    diagnostico = encontrar_diagnostico_bert(texto_usuario, train)
    diagnostico_lower = diagnostico.lower()  # Padroniza para minúsculo
    especialista = especialistas.get(diagnostico_lower, "Especialista não encontrado")

    print(f"\nPossível diagnóstico: {diagnostico}")
    print(f"Especialista recomendado: {especialista}")

if __name__ == "__main__":
    main()  # Executa o programa principal se o script for chamado diretamente