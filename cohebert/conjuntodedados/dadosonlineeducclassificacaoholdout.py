# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadosonlineeducmedida import *

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
    Organiza osdados do OnlineEduc 1.0 para classificação e retorna um dataframe.
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
def getConjuntoDeDadosClassificacao(model_args, tokenizer): 
    '''    
    Carrega os dados do OnlineEduc 1.0 para classificação e retorna um dataframe.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.  
        `tokenizer` - Tokenizador BERT.
    Saída:
        `dfdados` - Um dataframe com os dados carregados.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados()
        
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosOnlineEduc()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosGrandes(model_args, tokenizer, dfdados)
    
    # Organiza os dados para classificação
    dfdados = organizaDados(dfdados)
    
    return dfdados  
