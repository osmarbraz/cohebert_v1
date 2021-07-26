# Import das bibliotecas.
import logging  # Biblioteca de logging
import os
import pandas as pd

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
            logging.info('Arquivo com as medições não encontrado')        

    else:
        logging.info('Diretório com as medições não encontrado')

    return dfMedida
