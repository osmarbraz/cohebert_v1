# Import das bibliotecas.
import logging  # Biblioteca de logging
import pandas as pd # Biblioteca para manipulação e análise de dados
from sklearn.model_selection import train_test_split # Biblioteca de divisão

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadoscstnewsmedida import *
        
# ============================
def divisaoConjuntoDados(dfdados, percentualDivisao=0.3, classeStratify='classe'):
    '''    
    Divide o conjunto de dados em treino e teste utilizando um percentual de divisão.
    Parâmetros:
        `dfdados` - Dataframe com os dados a serem divididos.  
        `percentualDivisao` - Percentual de divisão dos dados.
        `classeStratify` - Faz uma divisão de forma que a proporção dos valores na amostra produzida seja a mesma que a proporção dos valores fornecidos.
    Saída:
        `dfdados_train` - Dataframe com os dados de treinamento.
        `dfdados_test` - Dataframe com os dados de teste.
    '''
        
    # Quantidade de elementos de teste considerando o percentual
    test_qtde = int(percentualDivisao*dfdados.shape[0])
    
    # Divide o conjunto
    dfdados_train, dfdados_test = train_test_split(dfdados, test_size=test_qtde, random_state=42, stratify=dfdados[classeStratify])
    
    logging.info("Conjunto total: {}.".format(len(dfdados)))
    logging.info("  Treino: {}.".format(len(dfdados_train)))
    logging.info("  Teste : {}.".format(len(dfdados_test)))

    return dfdados_train, dfdados_test

# ============================
def organizaDados(dfdados):
    '''
    Organiza osdados do CSTNews para classificação e retorna um dataframe. 
    Coloca os dados dos pares de documento um após o outro. 
    Primeiro adiciona o original e rotula como 1 e depois coloca o permutado rotulando como 0.
    Parâmetros:
        `dfdados` - Dataframe com os dados a serem organizados para classificação.    
    '''
    
    # Organiza os dados
    dados_organizados = []

    # Coloca o par um embaixo do outro.
    for index, linha in dfdados.iterrows():        
        # 1 Para original
        dados_organizados.append((linha['idOriginal'],linha['documentoOriginal'],1))    
        # 0 para uma permutação 
        dados_organizados.append((linha['idPermutado'],linha['documentoPermutado'],0))

    # Cria um dataframe com os dados
    dfdados = pd.DataFrame(dados_organizados, columns=["id","documento","classe"])      
    
    return dfdados 
    
# ============================
def getConjuntoDeDadosClassificacao(model_args, tokenizer, ORIGEM):  
    '''    
    Carrega os dados do CSTNews e retorna um dataframe para classificação.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.            
        `tokenizer` - Tokenizador BERT.
        `ORIGEM` - Se a variável for setada indica de onde fazer o download.       
    Saída:
        `dfdados` - Um dataframe com os dados carregados.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados(ORIGEM)
    
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosCSTNews()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
    
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosGrandes(model_args, tokenizer, dfdados)
    
    # Organiza os dados para classificação
    dfdados = organizaDados(dfdados)
    
    return dfdados

# ============================
def descartandoDocumentosGrandesTreinoTeste(model_args, tokenizer, dfdados_train, dfdados_test):
    '''
    Descarta os documentos que possuem mais tokens que o tamanho máximo em model_args(max_seq_len). 
    Parâmetros:        
        `model_args` - Objeto com os argumentos do modelo.
        `tokenizer` - Tokenizador BERT.
        `dfdados_train` - Dataframe com os dados de treinamento.
        `dfdados_test` - Dataframe com os dados de teste.
    Saída:
        `dfdados_train` - Dataframe com os dados de treinamento sem documentos grandes.
        `dfdados_test` - Dataframe com os dados de teste sem documentos grandes.
    '''    
    
    # Verifica se o tokenizador foi carregado
    if tokenizer != None:
        
        # Define o tamanho máximo para os tokens
        tamanho_maximo = model_args.max_seq_len

        logging.info("Removendo documentos grandes, acima de {} tokens.".format(tamanho_maximo))

        # Tokenize a codifica as setenças para o BERT     
        dfdados_train['input_ids'] = dfdados_train['documento'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        dfdados_train = dfdados_train[dfdados_train['input_ids'].apply(len)<tamanho_maximo]

        logging.info("Tamanho do dataset de treino: {}.".format(len(dfdados_train)))

        # Remove as colunas desnecessárias
        dfdados_train = dfdados_train.drop(columns=['input_ids'])

        # Tokenize a codifica as setenças para o BERT.     
        dfdados_test['input_ids'] = dfdados_test['documento'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        # Corta os inputs para o tamanho máximo 512
        dfdados_test = dfdados_test[dfdados_test['input_ids'].apply(len)<tamanho_maximo]

        logging.info("Tamanho do dataset de teste: {}".format(len(dfdados_test)))

        # Remove as colunas desnecessárias
        dfdados_test = dfdados_test.drop(columns=['input_ids'])
    else:
        logging.info("Tokenizador não definido.")        

    return dfdados_train, dfdados_test
