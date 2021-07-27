# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca para manipulação e análise de dados
from sklearn.model_selection import train_test_split # Biblioteca de divisão
import numpy as np # Biblioteca para manipulação de arrays e matrizes multidimensionais
from sklearn.model_selection import KFold # Biblioteca de divisão Kfold

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadoscstnewsmedida import *

# ============================
def analiseArquivosKFold(model_args, DIRETORIO_BASE, tokenizer):
    '''    
    Analisa os dados dos arquivos para Kfolds.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.    
        `DIRETORIO_BASE` - Diretório onde salvar os dados.  
        `tokenizer` - Tokenizador BERT.        
    '''
  
    logging.info("Análise dos dados dos arquivos dos KFolds do diretório: {}.".format(DIRETORIO_BASE))

    # Lista para armazenar os dados
    lista_dadostrain_folds = []
    lista_dadostest_folds = []

    # Define o prefixo do nome dos arquivos dos folds
    PREFIXO_NOME_ARQUIVO_TREINO = DIRETORIO_BASE + "/cstnews_md_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = DIRETORIO_BASE + "/cstnews_md_test_f"

    # Quantidade de folds a ser gerado
    QTDE_FOLDS = model_args.fold

    for x in range(QTDE_FOLDS):
  
        dadostrain = pd.read_csv(PREFIXO_NOME_ARQUIVO_TREINO+str(x+1)+".csv", sep=';')
        logging.info("Dados treino do fold {}: {}.".format(x+1,len(dadostrain)))

        dadostest = pd.read_csv(PREFIXO_NOME_ARQUIVO_TESTE+str(x+1)+".csv", sep=';')
        logging.info("Dados teste do fold {}: {}.".format(x+1,len(dadostest)))

        lista_dadostrain_folds.append([x,dadostrain.tipo.sum(), len(dadostrain.tipo)-dadostrain.tipo.sum()])
        lista_dadostest_folds.append([x,dadostest.tipo.sum(), len(dadostest.tipo)-dadostest.tipo.sum()])

        # Pega as listas de documentos e seus rótulos.
        documentos = dadostrain.documento.values
        maior_tamanho_documento_treino = 0

        # Para cada documento no conjunto de treino.
        for documento in documentos:
            # Tokeniza o texto e adiciona os tokens `[CLS]` e `[SEP]`.
            input_ids = tokenizer.encode(documento, add_special_tokens=True)
            # Atualiza o tamanho máximo de documento.
            maior_tamanho_documento_treino = max(maior_tamanho_documento_treino, len(input_ids))
            
        print('Máximo de tokens no conjunto de dados de treino: {}'.format(maior_tamanho_documento_treino))

        # Pega as listas de documentos e seus rótulos.
        documentos = dadostest.documento.values
        maior_tamanho_documento_teste = 0    
        # Para cada documento no conjunto de treino.  
        for documento in documentos:
            # Tokeniza o texto e adiciona os tokens `[CLS]` e `[SEP]`.
            input_ids = tokenizer.encode(documento, add_special_tokens=True)
            # Atualiza o tamanho máximo de documento.
            maior_tamanho_documento_teste = max(maior_tamanho_documento_teste, len(input_ids))
            
        logging.info("Máximo de token no conjunto de dados de teste: {}".format(maior_tamanho_documento_teste))

        logging.info("Fold {} Treino positivos: {} of {} ({:.2f}%)".format(x+1, 
                                                                  dadostrain.tipo.sum(), 
                                                                  len(dadostrain.tipo), 
                                                                  (dadostrain.tipo.sum() / len(dadostrain.tipo) * 100.0)
                                                                  ))

        logging.info("Fold {} Treino negativos: {} of {} ({:.2f}%)".format(x+1, 
                                                                  len(dadostrain.tipo)-dadostrain.tipo.sum(), 
                                                                  len(dadostrain.tipo), 
                                                                  ((len(dadostrain.tipo)-dadostrain.tipo.sum()) / len(dadostrain.tipo) * 100.0)))

        logging.info("Fold {} Teste positivos: {} of {} ({:.2f}%)".format(x+1, 
                                                                  dadostest.tipo.sum(), 
                                                                  len(dadostest.tipo), 
                                                                  (dadostest.tipo.sum() / len(dadostest.tipo) * 100.0)))
        logging.info("Fold {} Teste negativos: {} of {} ({:.2f}%)".format(x+1, 
                                                                  len(dadostest.tipo)-dadostest.tipo.sum(), 
                                                                  len(dadostest.tipo), 
                                                                  ((len(dadostest.tipo)-dadostest.tipo.sum()) / len(dadostest.tipo) * 100.0)))                               
        logging.info("")

# ============================
def gerarArquivosKFold(model_args, DIRETORIO_BASE, dfdados):
    '''    
    Divide o conjunto de dados em arquivos de treino e teste para Kfolds.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.    
        `DIRETORIO_BASE` - Diretório onde salvar os dados.  
        `dfdados` - Dataframe com os dados a serem divididos.        
        
    Saída:
        Arquivos dos KFolds salvos no diretório base.
    '''

    # Cria o diretório para receber os folds    
    if not os.path.exists(DIRETORIO_BASE):  
        # Cria o diretório
        os.makedirs(DIRETORIO_BASE)    
        logging.info("Diretório criado: {}.".format(DIRETORIO_BASE))
    else:    
        logging.info("Diretório já existe: {}.".format(DIRETORIO_BASE))

    # Quantidade de folds a ser gerado
    QTDE_FOLDS = model_args.fold

    # Define o prefixo do nome dos arquivos dos folds
    PREFIXO_NOME_ARQUIVO_TREINO = DIRETORIO_BASE + "/cstnews_md_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = DIRETORIO_BASE + "/cstnews_md_test_f"

    # Preparação do conjunto de dados.
    X =  np.array(dfdados)

    # Divisão em k folds(n_splits).
    # shuffle, embaralha cada amostra.
    # Quando shuffle igual True, random_state afeta a ordem dos índices, que controla a aleatoriedade de cada dobra.
    kf = KFold(n_splits=QTDE_FOLDS,  random_state=True, shuffle=True)
    
    CONTAFOLD = 1
    
    # Percorre os indices do conjunto de dados.
    for train_index, test_index in kf.split(X):
      logging.info("Executando divisão do fold: {}, Total: {}".format(CONTAFOLD, len(train_index)+len(test_index)))
      logging.info("Treino: {}, Teste: {}".format(len(train_index), len(test_index)))

      #print("�?ndices de treino:", len(train_index), " - ", train_index[0], " - ", train_index[len(train_index)-1])  
      #print(train_index)
      #print("�?ndices de teste :", len(test_index), "  - ", test_index[0], " - ", test_index[len(test_index)-1])
      #print(test_index)

      # Recupera os dados das documentos para treino e teste.
      documentos_train = X[train_index]
      documentos_test  = X[test_index]  

      # Organiza dados de treino.
      documentos_train_organizada = []

      # Adiciona a documento incoerente logo após a coerente para treino.
      for linha in documentos_train:     
          # 1 para o documento original
          documentos_train_organizada.append((linha[0], linha[2], 1))  
          # 0 para o documento permutado  
          documentos_train_organizada.append((linha[3], linha[5], 0))
    
          # Cria um dataframe com os dados de treinamento.
          pddata_tuples_train = pd.DataFrame(documentos_train_organizada, columns=["id","documento","tipo"])
    
      # Salva o arquivo de treino do fold.
      pddata_tuples_train.to_csv(PREFIXO_NOME_ARQUIVO_TREINO + str(CONTAFOLD)+".csv", index = False, sep=';')

      # Organiza dados de teste.
      documentos_test_organizada = []

      # Adiciona a documento incoerente logo após a coerente para teste.
      for linha in documentos_test:    
          # 1 Para coerente.
          documentos_test_organizada.append((linha[0], linha[2], 1))
          # 0 para uma permutação como incoerente.
          documentos_test_organizada.append((linha[3], linha[5], 0))
    
      # Cria um dataframe com os dados de teste.
      pddata_tuples_test = pd.DataFrame(documentos_test_organizada, columns=["id","documento","tipo"])  
  
      # Salva o arquivo de teste do fold.
      pddata_tuples_test.to_csv(PREFIXO_NOME_ARQUIVO_TESTE+str(CONTAFOLD)+".csv", index = False, sep=';')

      # Avança o contador de testes.
      CONTAFOLD = CONTAFOLD + 1        

# ============================
def copiaCSTNewsGithub():  
    '''    
    Copia dos arquivos do conjunto de dados do CSTNews para classifica��o KFold do Github.
    '''
    
    logging.info("Copiando do CSTNews do checkout do Github")

    # Diret�rio dos arquivos de dados.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diret�rio existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diret�rio
        os.makedirs(DIRETORIO)
        logging.info("Diret�rio criado: {}.".format(DIRETORIO))
    else:
        logging.info("Diret�rio j� existe: {}.".format(DIRETORIO))
        
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = "CSTNEWS_MD_KFOLD_10.zip"
    
    # Diret�rios dos arquivos
    DIRETORIO_FONTE_ARQUIVO = "/content/cohebert_v1/conjuntodedados/cstnews/" + NOME_ARQUIVO
    DIRETORIO_DESTINO_ARQUIVO = "/content/" + NOME_ARQUIVO
    
    # Apaga o arquivo    
    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)
    
    # Copia o arquivo de dados do diret�rio fonte para o diret�rio de destino
    shutil.copy(DIRETORIO_FONTE_ARQUIVO, DIRETORIO_DESTINO_ARQUIVO) 
        
    # Descompacta o arquivo na pasta de descompacta��o.                
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO,"r")
    arquivoZip.extractall(DIRETORIO)     

# ============================
def downloadCSTNewsGithub():  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews para classificação KFold do Github.
    '''

    logging.info("Download do CSTNews do Github")  

    # Diretório dos arquivos de dados
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório
        os.makedirs(DIRETORIO)
        logging.info('Diretório criado: {}'.format(DIRETORIO))
    else:
        logging.info('Diretório já existe: {}'.format(DIRETORIO))
        
    # Download do arquivo de dados  
    
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'CSTNEWS_MD_KFOLD_10.zip'

    # Apaga o arquivo.    
    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)
    
    # Realiza o download do arquivo do OneDrive
    URL_ARQUIVO = 'https://github.com/osmarbraz/cohebert_v1/blob/main/conjuntodedados/'+ NOME_ARQUIVO + '?raw=true'

    # Realiza o download do arquivo do conjunto de dados    
    downloadArquivo(URL_ARQUIVO, NOME_ARQUIVO)
    
    # Descompacta o arquivo na pasta de descompactação                
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO,"r")
    arquivoZip.extractall(DIRETORIO)       

# ============================
def getConjuntoDeDadosClassificacao(model_args, ORIGEM, tokenizer):  
    '''    
    Carrega os dados do CSTNews e retorna um dataframe para classificação.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.    
        `ORIGEM` - Se a variável for setada indica de onde fazer o download.       
        `tokenizer` - Tokenizador BERT.
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
def downloadCSTNews(ORIGEM):
    '''
    Realiza o download o arquivo KFold do CSTNews de uma determiada origem(ORIGEM).
    Parâmetros:
        `ORIGEM` - Se a variável for setada indica para fazer o download do Github caso contrário usar a copia do checkout.       
    '''    
    
    if ORIGEM:
        # Realiza o download do conjunto de dados dos folds
        downloadCSTNewsGithub()
    else:
        # Copia do diretório do github do checkout
        copiaCSTNewsGithub()
            
# ============================
def getConjuntoDeDadosClassificacao(model_args, tokenizer, ORIGEM):
    '''    
    Carrega os dados do CSTNews de um fold e retorna um dataframe para classificação.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.  
        `tokenizer` -Tokenizador do BERT para descartar documentos grandes.  
        `ORIGEM` - Se a variável for setada indica para fazer o download do Github caso contrário usar a copia do checkout.    
    Saída:
        `dfdados_train` - Dataframe com os dados de treinamento.
        `dfdados_test` - Dataframe com os dados de teste.
    '''
    
    # Fold de dados a ser carregado
    fold = model_args.fold
    
    # Diretório dos arquivos de dados dos folds.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if os.path.exists(DIRETORIO) == False:
        downloadCSTNews(ORIGEM)
  
    # Define o prefixo do nome dos arquivos dos folds
    PREFIXO_NOME_ARQUIVO_TREINO = "cstnews_md_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = "cstnews_md_test_f"

    # Nome dos arquivos.
    ARQUIVO_TREINO = DIRETORIO + "/" + PREFIXO_NOME_ARQUIVO_TREINO + str(fold) + ".csv"
    ARQUIVO_TESTE = DIRETORIO + "/" + PREFIXO_NOME_ARQUIVO_TESTE + str(fold) + ".csv" 

    logging.info("Carregando arquivo de treino: {}".format(ARQUIVO_TREINO))
    logging.info("Carregando arquivo de teste: {}".format(ARQUIVO_TESTE))

    # Carrega o dataset de treino e teste.
    dfdados_train = pd.read_csv(ARQUIVO_TREINO, sep=';')
    logging.info('Qtde de dados de treino: {}'.format(len(dfdados_train)))
    dfdados_test = pd.read_csv(ARQUIVO_TESTE, sep=';')
    logging.info('Qtde de dados de teste: {}'.format(len(dfdados_test)))

    # Remove os documentos muito grandes
    dfdados_train, dfdados_test = descartandoDocumentosGrandes(tokenizer, model_args, dfdados_train, dfdados_test)
    
    return dfdados_train, dfdados_test        
