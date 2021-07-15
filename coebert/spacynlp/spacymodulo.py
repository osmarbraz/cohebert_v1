# Import das bibliotecas.
import requests # Biblioteca para download
import tarfile # Biblioteca de descompactação
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos
import spacy # Biblioteca do spaCy

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

def downloadSpacy(model_args):
    '''
    Realiza o download do arquivo do modelo para o diretório corrente
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy
    NOME_ARQUIVO_MODELO_COMPACTADO = ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"
    
    # Url do arquivo
    URL_ARQUIVO_MODELO_COMPACTADO = "https://github.com/explosion/spacy-models/releases/download/" + ARQUIVOMODELOSPACY + VERSAOSPACY + "/" + NOME_ARQUIVO_MODELO_COMPACTADO

    # Realiza o download do arquivo do modelo
    downloadArquivo(URL_ARQUIVO_MODELO_COMPACTADO, NOME_ARQUIVO_MODELO_COMPACTADO)

def descompactaSpacy(model_args):
    '''
    Descompacta o arquivo do modelo
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy
    
    # Nome do arquivo a ser descompactado
    NOME_ARQUIVO_MODELO_COMPACTADO = ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"
    
    arquivoTar = tarfile.open(NOME_ARQUIVO_MODELO_COMPACTADO, "r:gz")    
    arquivoTar.extractall()    
    arquivoTar.close()
    
    # Apaga o arquivo compactado
    if os.path.isfile(NOME_ARQUIVO_MODELO_COMPACTADO):
        os.remove(NOME_ARQUIVO_MODELO_COMPACTADO)
    
def moveSpacy(model_args):
    '''
    Coloca a pasta do modelo descompactado em uma pasta de nome mais simples.
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy
    
    # Caminho da origem do diretório
    ORIGEM = "/content/" + ARQUIVOMODELOSPACY + VERSAOSPACY + "/" + ARQUIVOMODELOSPACY + "/" + ARQUIVOMODELOSPACY + VERSAOSPACY 
    # Destino do diretório
    DESTINO = "/content/" + ARQUIVOMODELOSPACY

    shutil.move(ORIGEM, DESTINO)
    
def carregaSpacy(model_args):
    '''
    Realiza o carregamento do Spacy.    
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy

    downloadSpacy(model_args)

    descompactaSpacy(model_args)
    
    moveSpacy(model_args)

    # Caminho completo do modelo do spaCy
    CAMINHOMODELOSPACY = '/content/' + ARQUIVOMODELOSPACY

    # Necessário 'tagger' para encontrar os substantivos
    nlp = spacy.load(CAMINHOMODELOSPACY, disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])

    return nlp

def getStopwords(nlp):
    '''
    Recupera as stop words do nlp(Spacy).
    '''
    
    spacy_stopwords = nlp.Defaults.stop_words

    return spacy_stopwords 
