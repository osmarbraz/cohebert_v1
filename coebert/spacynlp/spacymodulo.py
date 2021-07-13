# Import das bibliotecas.
import wget
import tarfile
import shutil

def downloadSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Realiza o download do arquivo do modelo para o diretório corrente
    '''
    # Url do arquivo
    URL_ARQUIVO = "https://github.com/explosion/spacy-models/releases/download/" + ARQUIVOMODELOSPACY + VERSAOSPACY + "/" + ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"

    # Realiza o download do arquivo
    wget.download(URL_ARQUIVO)        

def descompactaSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Descompacta o arquivo do modelo
    '''
    # Nome do arquivo a ser descompactado
    ARQUIVO = ARQUIVOMODELOSPACY + VERSAOSPACY + ".tar.gz"
    
    arquivoTar = tarfile.open(ARQUIVO, "r:gz")    
    arquivoTar.extractall()    
    arquivoTar.close()    
    
def moveSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Coloca a pasta do modelo descompactado em uma pasta de nome mais simples.
    '''
    
    ORIGEM = "/content/" + ARQUIVOMODELOSPACY + VERSAOSPACY + "/" + ARQUIVOMODELOSPACY + "/" + ARQUIVOMODELOSPACY + VERSAOSPACY 
    DESTINO = "/content/" + ARQUIVOMODELOSPACY

    shutil.move(ORIGEM, DESTINO)
    
def carregaSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Realiza o carregamento do Spacy.    
    '''

    downloadSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY)

    descompactaSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY)
    
    moveSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY)

    # Importando as bibliotecas.
    import spacy

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
