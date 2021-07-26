# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas
from sklearn.model_selection import train_test_split # Biblioteca de divisão
import numpy as np
from sklearn.model_selection import KFold

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadoscstnewsmedida import *


# ============================
def gerarArquivosKFoldCSTNews(DIRETORIO_BASE, dfdados, model_args)
    '''    
    Divide o conjunto de dados em Kfolds.
    Parâmetros:
        `DIRETORIO_BASE` - Diretório onnde salvar os dados.  
        `dfdados` - Dataframe com os dados a serem divididos. 
        `model_args` - Objeto com os argumentos do modelo.    
        
    Saída:
        Arquivos dos KFolds salvos no diretório base.
    '''

    # Cria o diretório para receber os folds    
    if not os.path.exists(DIRETORIO_BASE):  
        # Cria o diretório
        os.makedirs(DIRETORIO_BASE)    
        logging.info('Diretório criado: {}'.format(DIRETORIO_BASE))
    else:    
        logging.info('Diretório já existe: {}'.format(DIRETORIO_BASE))

    # Quantidade de folds a ser gerado
    QTDE_FOLDS = model_args.fold

    # Define o prefixo do nome dos arquivos dos folds
    PREFIXO_NOME_ARQUIVO_TREINO = DIRETORIO_BASE + "/cstnews_md_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = DIRETORIO_BASE + "/cstnews_md_test_f"

    # Registra o tempo inícial.
    t0 = time.time()

    # Preparação do conjunto de dados.
    X =  np.array(dfdados)

    # Divisão em k folds(n_splits).
    # shuffle, embaralha cada amostra.
    # Quando shuffle igual True, random_state afeta a ordem dos índices, que controla a aleatoriedade de cada dobra.
    kf = KFold(n_splits=QTDE_FOLDS,  random_state=True, shuffle=True)
    
    CONTAFOLD = 1
    # Percorre os indices do conjunto de dados.
    for train_index, test_index in kf.split(X):
      logging.info("\nExecutando divisão do fold: {}, Total: {}".format(CONTAFOLD, len(train_index)+len(test_index)))
      logging.info("Treino: {}, Teste: {}".format(len(train_index), len(test_index)))

      #print("Índices de treino:", len(train_index), " - ", train_index[0], " - ", train_index[len(train_index)-1])  
      #print(train_index)
      #print("Índices de teste :", len(test_index), "  - ", test_index[0], " - ", test_index[len(test_index)-1])
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
      pddata_tuples_test.to_csv(PREFIXO_NOME_ARQUIVO_TESTE+str(TESTE)+".csv", index = False, sep=';')

      # Avança o contador de testes.
      CONTAFOLD = CONTAFOLD + 1

      # Medida de quanto tempo levou a execução da validação.
      teste_time = formataTempo(time.time() - t0)

      logging.info("  Tempo gasto: {:}".format(teste_time))


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
def downloadCSTNewsKFoldGithub():  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews para classificação KFold do Github.
    '''

    logging.info("Download do Github")  

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
def copiaCSTNewsKFoldGithub():  
    '''    
    Copia dos arquivos do conjunto de dados do CSTNews para classificação KFold do Github.
    '''
    
    logging.info("Copiando do checkout do Github")

    # Diretório dos arquivos de dados.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório
        os.makedirs(DIRETORIO)
        logging.info('Diretório criado: {}'.format(DIRETORIO))
    else:
        logging.info('Diretório já existe: {}'.format(DIRETORIO))
        
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = "CSTNEWS_MD_KFOLD_10.zip"
    
    # Diretórios dos arquivos
    DIRETORIO_FONTE_ARQUIVO = "/content/cohebert_v1/conjuntodedados/cstnews/" + NOME_ARQUIVO
    DIRETORIO_DESTINO_ARQUIVO = "/content/" + NOME_ARQUIVO
    
    # Apaga o arquivo    
    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)
    
    # Copia o arquivo de dados do diretório fonte para o diretório de destino
    shutil.copy(DIRETORIO_FONTE_ARQUIVO, DIRETORIO_DESTINO_ARQUIVO) 
        
    # Descompacta o arquivo na pasta de descompactação.                
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
    dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    
    # Organiza os dados para classificação
    dfdados = organizaDados(dfdados)
    
    return dfdados

# ============================
def descartandoDocumentosGrandes(tokenizer, model_args, dfdados_train, dfdados_test):
    '''
    Descarta os documentos que possuem mais tokens que o tamanho máximo em model_args(max_seq_len). 
    Parâmetros:
        `tokenizer` - Tokenizador BERT.
        `model_args` - Objeto com os argumentos do modelo.
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

# ============================
def downloadCSTNewsKFold(ORIGEM):
    '''
    Realiza o download o arquivo KFold do CSTNews de uma determiada origem(ORIGEM).
    Parâmetros:
        `ORIGEM` - Se a variável for setada indica para fazer o download do Github caso contrário usar a copia do checkout.       
    '''    
    
    if ORIGEM:
        # Realiza o download do conjunto de dados dos folds
        downloadCSTNewsKFoldGithub()
    else:
        # Copia do diretório do github do checkout
        copiaCSTNewsKFoldGithub()
            
# ============================
def getConjuntoDeDadosClassificacaoKFold(model_args, tokenizer, ORIGEM):
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
        downloadCSTNewsKFold(ORIGEM)
  
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
