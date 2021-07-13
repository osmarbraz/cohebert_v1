def downloadSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Realiza o download do arquivo do modelo para o diretório corrente
    '''
    !wget https://github.com/explosion/spacy-models/releases/download/{ARQUIVOMODELOSPACY}{VERSAOSPACY}/{ARQUIVOMODELOSPACY}{VERSAOSPACY}.tar.gz

def descompactaSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Descompacta o arquivo do modelo
    '''
    !tar -xvf  /content/{ARQUIVOMODELOSPACY}{VERSAOSPACY}.tar.gz

def moveSpacy(ARQUIVOMODELOSPACY, VERSAOSPACY):
    '''
    Coloca a pasta do modelo descompactado em uma pasta de nome mais simples
    '''
    !mv /content/{ARQUIVOMODELOSPACY}{VERSAOSPACY}/{ARQUIVOMODELOSPACY}/{ARQUIVOMODELOSPACY}{VERSAOSPACY} /content/{ARQUIVOMODELOSPACY}


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
