# Import das bibliotecas.
import os
import pandas as pd

def carregaMedidasCSTNews(CAMINHO_MEDIDAS, TIPO_MODELO, ESTRATEGIA_POOLING, PALAVRA_RELEVANTE, NOME_MODELO_BERT, TAMANHO_BERT):
  '''
  Carrega as medidas de coerência de um diretório e retorna um dataframe.
  '''
  
  NOME_BASE = 'MedicaoCoerenciaCSTNews_v1'

  NOME_ARQUIVO_MEDICAO = NOME_BASE + TIPO_MODELO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + NOME_MODELO_BERT + TAMANHO_BERT + '.csv'
                                             
  # Verifica se o diretório dos resultados existem.
  if os.path.exists(CAMINHO_MEDIDAS):
      arquivos = os.listdir(CAMINHO_MEDIDAS)     

      NOME_ARQUIVO_MEDICAO_COMPLETO = CAMINHO_MEDIDAS + NOME_ARQUIVO_MEDICAO
    
      # Verifica se o arquivo existe.
      if os.path.isfile(NOME_ARQUIVO_MEDICAO_COMPLETO):
          print("Carregando arquivo:", NOME_ARQUIVO_MEDICAO)
          # Carrega os dados do arquivo  
          dfMedida = pd.read_csv(NOME_ARQUIVO_MEDICAO_COMPLETO, sep=';')
      
      else:
          print('Arquivo com as medições não encontrado')        

  else:
      print('Diretório com as medições não encontrado')

  print('Medidas carregadas: ', len(dfMedida))

  return dfMedida
