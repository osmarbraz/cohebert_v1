# Import das bibliotecas.
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadosonlineducmedida import *

def divisaoConjuntoDados(dfdados, percentualDivisao=0.3, classeStratify='classe'):
    '''    
    Divide o conjunto de dados em treino e teste utilizando um percentual de divisão.
    '''
        
    # Quantidade de elementos de teste considerando o percentual
    test_qtde = int(percentualDivisao*dfdados.shape[0])
    
    # Divide o conjunto
    dfdados_train, dfdados_test = train_test_split(dfdados, test_size=test_qtde, random_state=42, stratify=dfdados[classeStratify])

    print("Conjunto total:", len(dfdados))
    print("  Treino:", len(dfdados_train))
    print("  Teste :", len(dfdados_test))

    return dfdados_train, dfdados_test

  def organizaDados(dfdados):
    '''    
    Organiza osdados do CSTNews para classificação e retorna um dataframe.
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
  
def getConjuntoDeDadosClassificacao(model_args, tokenizer): 
    '''    
    Carrega os dados do OnlineEduc 1.0 para classificação e retorna um dataframe.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados()
        
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosOnlineEduc()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    
    # Organiza os dados para classificação
    dfdados = organizaDados(dfdados)
    
    return dfdados  
