# Import das bibliotecas.
import wget # Biblioteca de download
import tarfile # Biblioteca de descompactação
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos
import spacy # Biblioteca do spaCy
    
def downloadSpacy(model_args):
    '''
    Realiza o download do arquivo do modelo para o diretório corrente
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy
    
    # Url do arquivo
    URL_ARQUIVO_MODELO = "https://github.com/explosion/spacy-models/releases/download/" + ARQUIVOMODELOSPACY + VERSAOSPACY + "/" + ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"

    # Realiza o download do arquivo do modelo
    wget.download(URL_ARQUIVO_MODELO)        

def descompactaSpacy(model_args):
    '''
    Descompacta o arquivo do modelo
    '''
    
    ARQUIVOMODELOSPACY = model_args.modelo_spacy
    VERSAOSPACY = "-" + model_args.versao_spacy
    
    # Nome do arquivo a ser descompactado
    ARQUIVO = ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"
    
    arquivoTar = tarfile.open(ARQUIVO, "r:gz")    
    arquivoTar.extractall()    
    arquivoTar.close()
    
    # Apaga o arquivo compactado
    if os.path.isfile(ARQUIVO)
        os.remove(ARQUIVO)
    
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
