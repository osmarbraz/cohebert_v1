# Import das bibliotecas.
import logging  # Biblioteca de logging
import os
import pandas as pd

# ============================
def organizaParesDocumentosCSTNews(dfOriginalMedida, dfPermutadoMedida):
    '''
    Organiza as medidas do.
    Parâmetros:
        `dfOriginalMedida` - Dados de medidas de documentos originais.
        `dfPermutadoMedida` - Dados de medidas de documentos permutados.
        
    Saída:
        `dfListaParesDocumentosMedidas` - Dataframe com as medidas de documentos originais seguidos das medidas de documentos permutados.
    '''  

    listaParesDocumentosMedidas = []
    # Refaz os pares de documentos
    for i, linha1 in dfOriginalMedida.iterrows():
        ponto = linha1['arquivo'].find('.')
        nomeArquivo = linha1['arquivo'][:ponto]
        for i, linha2 in dfPermutadoMedida.iterrows():
            if nomeArquivo in linha2['arquivo']:
                listaParesDocumentosMedidas.append(
                                                    [linha1['data'], 
                                                    linha1['arquivo'],	
                                                    linha1['ccos'], 
                                                    linha1['ceuc'], 
                                                    linha1['cman'],
                                       
                                                    linha2['data'], 
                                                    linha2['arquivo'],	
                                                    linha2['ccos'], 
                                                    linha2['ceuc'], 
                                                    linha2['cman'] ],)

    logging.info("Registros antes:", len(listaParesDocumentosMedidas))

    dfListaParesDocumentosMedidas = pd.DataFrame(listaParesDocumentosMedidas, columns=('dataDO', 'idDO', 'ccosDO', 'ceucDO', 'cmanDO', 'dataPerm', 'idPerm', 'ccosPerm', 'ceucPerm', 'cmanPerm'))   
    logging.info("Registros depois:", len(dfListaParesDocumentosMedidas))

    return dfListaParesDocumentosMedidas

# ============================
def separaDocumentosCSTNews(dadosMedida):
    '''
    Separa os dados do dataframe em originais e permutados.
    Parâmetros:
        `dadosMedida` - Dados a serem separados em originais e permutados.    
        
    Saída:
        `dfOriginalMedida` - Dataframe com os dados de documentos originais.        
        `dfPermutadoMedida` - Dataframe com os dados de documentos permutados.
    '''        
    # Separa os originais
    dfOriginalMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm')==False]
    # Remove os duplicados
    dfOriginalMedida = dfOriginalMedida.drop_duplicates(subset=['arquivo'])
    logging.info("Registros: ", len(dfOriginalMedida))

    # Separa os permutados
    dfPermutadoMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm')==True]
    # Remove os duplicados
    dfPermutadoMedida = dfPermutadoMedida.drop_duplicates(subset=['arquivo'])
    
    logging.info("Registros: ", len(dfPermutadoMedida))

    return dfOriginalMedida, dfPermutadoMedida

# ============================
def carregaMedidasCSTNews(DIRETORIO_MEDIDAS, TIPO_MODELO, ESTRATEGIA_POOLING, PALAVRA_RELEVANTE, NOME_MODELO_BERT, TAMANHO_BERT):
    '''
    Carrega as medidas de coerência de um diretório e retorna um dataframe.
    Parâmetros:
        `DIRETORIO_MEDIDAS` - Diretório com os arquivos das medidas.    
        `TIPO_MODELO` - Tipo do modelo(pretreinado ou ajustado) a ser carregado.  
        `ESTRATEGIA_POOLING` - Nome da estratégia de pooling(MEAN ou MAX).
        `PALAVRA_RELEVANTE` - Nome da estratégia de relevância(ALL, CLEAN ou NOUN).
        `NOME_MODELO_BERT` - Nome do modelo(BERTimbau ou BERT) a ser carregado.  
        `TAMANHO_BERT` - Tamanho do modelo(Base ou Large) a ser carregado. 
        
    Saída:
        `dfMedida` - Um dataframe com os dados carregados.
    '''
    NOME_BASE = "MedicaoCoerenciaCSTNews_v1"
    
    NOME_ARQUIVO_MEDICAO = NOME_BASE + TIPO_MODELO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + NOME_MODELO_BERT + TAMANHO_BERT + '.csv'

    dfMedida = None
     
    # Verifica se o diretório dos resultados existem.
    if os.path.exists(DIRETORIO_MEDIDAS):
        arquivos = os.listdir(DIRETORIO_MEDIDAS)     

        NOME_ARQUIVO_MEDICAO_COMPLETO = DIRETORIO_MEDIDAS + NOME_ARQUIVO_MEDICAO
    
        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_MEDICAO_COMPLETO):
            logging.info("Carregando arquivo:", NOME_ARQUIVO_MEDICAO)
            
            # Carrega os dados do arquivo  
            dfMedida = pd.read_csv(NOME_ARQUIVO_MEDICAO_COMPLETO, sep=';')
            
            logging.info('Medidas carregadas: ', len(dfMedida))
      
        else:
            logging.info('Arquivo com as medições não encontrado!')        

    else:
        logging.info('Diretório com as medições não encontrado!')

    return dfMedida
