# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca para manipular arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca para manipulação e análise de dados
import numpy as np # Biblioteca para manipulação de arrays e matrizes multidimensionais
from sklearn.model_selection import KFold # Biblioteca de divisão Kfold

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadosonlineeducmedida import *

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
    PREFIXO_NOME_ARQUIVO_TREINO = DIRETORIO_BASE + "/moodle_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = DIRETORIO_BASE + "/moodle_test_f"

    # Quantidade de folds a ser gerado
    QTDE_FOLDS = model_args.fold

    for x in range(QTDE_FOLDS):
  
        dadostrain = pd.read_csv(PREFIXO_NOME_ARQUIVO_TREINO + str(x + 1) + ".csv", sep=';')
        logging.info("Dados treino do fold {}: {}.".format(x + 1, len(dadostrain)))

        dadostest = pd.read_csv(PREFIXO_NOME_ARQUIVO_TESTE + str(x + 1) + ".csv", sep=';')
        logging.info("Dados teste do fold {}: {}.".format(x + 1, len(dadostest)))

        lista_dadostrain_folds.append([x, dadostrain.tipo.sum(), len(dadostrain.tipo) - dadostrain.tipo.sum()])
        lista_dadostest_folds.append([x, dadostest.tipo.sum(), len(dadostest.tipo) - dadostest.tipo.sum()])

        # Pega as listas de documentos e seus rótulos.
        documentos = dadostrain.documento.values
        maior_tamanho_documento_treino = 0

        # Para cada documento no conjunto de treino.
        for documento in documentos:
            # Tokeniza o texto e adiciona os tokens `[CLS]` e `[SEP]`.
            input_ids = tokenizer.encode(documento, add_special_tokens=True)
            # Atualiza o tamanho máximo de documento.
            maior_tamanho_documento_treino = max(maior_tamanho_documento_treino, len(input_ids))
            
        print("Máximo de tokens no conjunto de dados de treino: {}.".format(maior_tamanho_documento_treino))

        # Pega as listas de documentos e seus rótulos.
        documentos = dadostest.documento.values
        maior_tamanho_documento_teste = 0    
        # Para cada documento no conjunto de treino.  
        for documento in documentos:
            # Tokeniza o texto e adiciona os tokens `[CLS]` e `[SEP]`.
            input_ids = tokenizer.encode(documento, add_special_tokens=True)
            # Atualiza o tamanho máximo de documento.
            maior_tamanho_documento_teste = max(maior_tamanho_documento_teste, len(input_ids))
            
        logging.info("Máximo de token no conjunto de dados de teste: {}.".format(maior_tamanho_documento_teste))

        logging.info("Fold {} Treino positivos: {} de {} ({:.2f}%).".format(x + 1, 
                     dadostrain.tipo.sum(), 
                     len(dadostrain.tipo), 
                     (dadostrain.tipo.sum() / len(dadostrain.tipo) * 100.0)
                     ))

        logging.info("Fold {} Treino negativos: {} de {} ({:.2f}%).".format(x + 1, 
                     len(dadostrain.tipo)-dadostrain.tipo.sum(), 
                     len(dadostrain.tipo), 
                     ((len(dadostrain.tipo)-dadostrain.tipo.sum()) / len(dadostrain.tipo) * 100.0)))

        logging.info("Fold {} Teste positivos: {} de {} ({:.2f}%).".format(x + 1, 
                     dadostest.tipo.sum(), 
                     len(dadostest.tipo), 
                     (dadostest.tipo.sum() / len(dadostest.tipo) * 100.0)))
        
        logging.info("Fold {} Teste negativos: {} de {} ({:.2f}%).".format(x + 1, 
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
        Atributos de dfdados:
        0. 'idOriginal' - Nome do arquivo original.
        1. 'sentencasOriginais' - Lista das sentenças do documento original.
        2. 'documentoOriginal' - Documento original.
        3. 'idPermutado' - Nome do arquivo permutado.
        4. 'sentencasPermutadas' - Lista das sentenças do documento permtuado.
        5. 'documentoPermutado' - Documento permutado. 
        
    Retorno:
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
    PREFIXO_NOME_ARQUIVO_TREINO = DIRETORIO_BASE + "/moodle_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = DIRETORIO_BASE + "/moodle_test_f"

    # Remove colunas desnecessárias
    dfdados = dfdados.drop(columns=['sentencasOriginais', 'sentencasPermutadas'])

    # Preparação do conjunto de dados.
    X = np.array(dfdados)

    # Divisão em k folds(n_splits).
    # shuffle, embaralha cada amostra.
    # Quando shuffle igual True, random_state afeta a ordem dos índices, que controla a aleatoriedade de cada dobra.
    kf = KFold(n_splits=QTDE_FOLDS, random_state=True, shuffle=True)
    
    CONTAFOLD = 1
    
    # Percorre os indices do conjunto de dados.
    for train_index, test_index in kf.split(X):
        logging.info("Executando divisão do fold: {}, Total: {}".format(CONTAFOLD, len(train_index) + len(test_index)))
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
            documentos_train_organizada.append((linha[2], linha[3], 0))
    
        # Cria um dataframe com os dados de treinamento.
        pddata_tuples_train = pd.DataFrame(documentos_train_organizada, columns=["id", "documento", "tipo"])
    
        # Salva o arquivo de treino do fold.
        pddata_tuples_train.to_csv(PREFIXO_NOME_ARQUIVO_TREINO + str(CONTAFOLD) + ".csv", index=False, sep=';')

        # Organiza dados de teste.
        documentos_test_organizada = []

        # Adiciona a documento incoerente logo após a coerente para teste.
        for linha in documentos_test:    
            # 1 Para coerente.
            documentos_test_organizada.append((linha[0], linha[1], 1))
            # 0 para uma permutação como incoerente.
            documentos_test_organizada.append((linha[2], linha[3], 0))
    
        # Cria um dataframe com os dados de teste.
        pddata_tuples_test = pd.DataFrame(documentos_test_organizada, columns=["id", "documento", "tipo"])  
  
        # Salva o arquivo de teste do fold.
        pddata_tuples_test.to_csv(PREFIXO_NOME_ARQUIVO_TESTE + str(CONTAFOLD) + ".csv", index=False, sep=';')

        # Avança o contador de testes.
        CONTAFOLD = CONTAFOLD + 1        

        
# ============================
def copiaOnlineEducGithub():  
    '''
    DADOS PRIVADOS, ARQUIVO DE DADOS NÃO DISPONÍVEL.
    Copia dos arquivos do conjunto de dados do OnlineEduc para classificação KFold do Github.
    '''
    
    logging.info("Copiando do OnlineEduc do checkout do Github")

    # Diretório dos arquivos de dados.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório
        os.makedirs(DIRETORIO)
        logging.info("Diretório para receber os dados criado: {}.".format(DIRETORIO))
    else:
        logging.info("Diretório para receber os dados já existe: {}.".format(DIRETORIO))
        
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = "MOODLE_KFOLD_10.zip"
    
    # Diretórios dos arquivos
    DIRETORIO_FONTE_ARQUIVO = "./conjuntodedados/onlineeduc1.0/" + NOME_ARQUIVO
    DIRETORIO_DESTINO_ARQUIVO = "/content/" + NOME_ARQUIVO
    
    # Apaga o arquivo    
    if os.path.isfile(NOME_ARQUIVO):
        os.remove(NOME_ARQUIVO)
    
    # Copia o arquivo de dados do diretório fonte para o diretório de destino
    shutil.copy(DIRETORIO_FONTE_ARQUIVO, DIRETORIO_DESTINO_ARQUIVO) 
        
    # Descompacta o arquivo no diretório de descompactação.                
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO, "r")
    arquivoZip.extractall(DIRETORIO)
    
    logging.info("Diretório com ")

# ============================
def downloadOnlineEducGithub():  
    '''
    DADOS PRIVADOS, ARQUIVO DE DADOS NÃO DISPONÍVEL.
    Download dos arquivos do conjunto de dados do OnlineEduc para classificação KFold do Github.
    '''

    logging.info("Download do OnlineEduc do Github")  
    
    # Verifica se existe o diretório base do cohebert e retorna o nome do diretório
    DIRETORIO_COHEBERT = verificaDiretorioCoheBERT()
   
    # Diretório dos arquivos de dados
    DIRETORIO = "/content/validacao_kfold"

    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório para receber os arquivos de dados
        os.makedirs(DIRETORIO)
        logging.info("Diretório para receber os dados criado: {}.".format(DIRETORIO))
    else:
        logging.info("Diretório para receber os dados já existe: {}.".format(DIRETORIO))

    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = "MOODLE_KFOLD_10.zip"

    # Apaga o arquivo.    
    if os.path.isfile(NOME_ARQUIVO):
        os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do OneDrive
    URL_ARQUIVO = "https://github.com/osmarbraz/" +  DIRETORIO_COHEBERT + "/blob/main/conjuntodedados/onlineeduc1.0/" + NOME_ARQUIVO + "?raw=true"

    # Realiza o download do arquivo do conjunto de dados    
    downloadArquivo(URL_ARQUIVO, NOME_ARQUIVO)

    # Descompacta o arquivo no diretório de descompactação                
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO, "r")
    arquivoZip.extractall(DIRETORIO)
  
# ============================
def copiaOnlineEducGoogleDrive(): 
    '''    
    Copia dos arquivos do conjunto de dados do OnlineEduc para classificação KFold do Google Drive.
    '''
    
    logging.info("Copia do OnlineEduc do Gooogle Drive")

    # Diretório do Google Drive
    DIRETORIO_GOOGLEDRIVE = "/content/drive"

    # Verifica se o diretório do google drive foi montada
    if os.path.exists(DIRETORIO_GOOGLEDRIVE):  
    
        # Cria o diretório para receber os arquivos de dados
        # Diretório dos arquivos de dados.
        DIRETORIO = "/content/validacao_kfold"

        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO):  
            # Cria o diretório
            os.makedirs(DIRETORIO)
            logging.info("Diretório para receber os dados criado: {}.".format(DIRETORIO))
        else:
            logging.info("Diretório para receber os dados já existe: {}.".format(DIRETORIO))

        # Nome do arquivo a ser criado.
        NOME_ARQUIVO = "MOODLE_KFOLD_10.zip"        

        # Define o caminho e nome do arquivo de dados
        CAMINHO_ARQUIVO = "/content/drive/MyDrive/Colab Notebooks/Data/Moodle/dadosmoodle_documento_pergunta_sentenca_intervalo/validacao_classificacao/kfold/" + NOME_ARQUIVO

        # Copia o arquivo compactado do conjunto do diretório no Google Drive para o diretório kfold.
        shutil.copy(CAMINHO_ARQUIVO, DIRETORIO) 

        # Descompacta o arquivo no diretório de descompactação.                
        arquivoZip = zipfile.ZipFile(DIRETORIO + '/' + NOME_ARQUIVO, "r")
        arquivoZip.extractall(DIRETORIO)

    else:
        logging.info("Diretório do google drive não foi montado: {}.".format(DIRETORIO_GOOGLEDRIVE))
    
# ============================
def downloadOnlineEduc(ORIGEM):
    '''
    Realiza o download o arquivo KFold do CSTNews de uma determinada origem(ORIGEM).
    
    Parâmetros:
    `ORIGEM` - Se a variável for setada indica para fazer o download do Github caso contrório usar a copia do checkout.       
    '''    
    
    #if ORIGEM:
        # Realiza o download do conjunto de dados dos folds 
    #    downloadOnlineEducGithub()
    #else:
        #copiaOnlineEducGithub()
    
    # O arquivo é restrito e está disponível somente para o Google Drive 
    copiaOnlineEducGoogleDrive()

# ============================   
def descartandoDocumentosGrandesClassificacaoFold(model_args, tokenizer, dfdados):
    '''    
    Remove os documentos que extrapolam 512 tokens.
    Você pode definir o tamanho de documento que quiser no BERT, mas o modelo pré-treinado vem com um tamanho pré-definido. 
    No nosso caso vamos utilizar o modelo BERT, que tem 512 tokens de tamanho limite de documento. 
    O tokenizador gera quantidades diferentes tokens para cada modelo pré-treinado. 
    Portanto é necessário especificar o tokenizador para descatar os documentos que ultrapassam o limite de tokens de entrada do BERT.
    
    Parâmetros:              
    `model_args` - Objeto com os argumentos do modelo.    
    `tokenizer` - Tokenizador BERT.
    `dfdados` - Dataframe com os documentos a serem analisados.   
    
    Retorno:
    `dfdadosretorno` - Um dataframe sem os documentos grandes.
    '''
    
    dfdadosretorno = None
    
    # Verifica se o tokenizador foi carregado
    if tokenizer != None:
    
        # Define o tamanho máximo para os tokens
        tamanho_maximo = model_args.max_seq_len
  
        # Tokenize a codifica os documentos para o BERT.     
        dfdados['input_ids'] = dfdados['documentoOriginal'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        # Reduz para o tamanho máximo suportado pelo BERT.
        dfdados_512 = dfdados[dfdados['input_ids'].apply(len) <= tamanho_maximo]

        # Remove as colunas desnecessárias.
        dfdadosAnterior = dfdados.drop(columns=['input_ids'])
        dfdadosretorno = dfdados_512.drop(columns=['input_ids'])

        logging.info("Quantidade de dados anterior: {}.".format(len(dfdadosAnterior)))
        logging.info("Nova quantidade de dados    : {}.".format(len(dfdadosretorno)))

        # Registros removidos
        df = dfdadosAnterior.merge(dfdadosretorno, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        logging.info("Quantidade de registros removidos: {}.".format(len(df)))
        
    else:
        logging.info("Tokenizador não definido.")        

    return dfdadosretorno              
            
# ============================
def getConjuntoDeDadosClassificacaoFold(model_args, tokenizer, ORIGEM):
    '''    
    Carrega os dados do OnlineEduc 1.0 de um fold e retorna um dataframe para classificação.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.  
    `tokenizer` - Tokenizador do BERT para descartar documentos grandes.  
    `ORIGEM` - Se a variável for setada indica para fazer o download do Github caso contrário usar a copia do checkout.
    
    Retorno:
    `dfdados_train` - Dataframe com os dados de treinamento.
    `dfdados_test` - Dataframe com os dados de teste.
    '''
    
    # Fold de dados a ser carregado
    fold = model_args.fold
    
    # Diretório dos arquivos de dados dos folds.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if os.path.exists(DIRETORIO) == False:
        downloadOnlineEduc(ORIGEM)
  
    # Define o prefixo do nome dos arquivos dos folds
    PREFIXO_NOME_ARQUIVO_TREINO = "moodle_train_f"
    PREFIXO_NOME_ARQUIVO_TESTE = "moodle_test_f"

    # Nome dos arquivos.
    ARQUIVO_TREINO = DIRETORIO + "/" + PREFIXO_NOME_ARQUIVO_TREINO + str(fold) + ".csv"
    ARQUIVO_TESTE = DIRETORIO + "/" + PREFIXO_NOME_ARQUIVO_TESTE + str(fold) + ".csv" 

    logging.info("Carregando arquivo de treino: {}'".format(ARQUIVO_TREINO))
    logging.info("Carregando arquivo de teste: {}'".format(ARQUIVO_TESTE))

    # Carrega o dataset de treino e teste.
    dfdados_train = pd.read_csv(ARQUIVO_TREINO, sep=';')
    logging.info("Qtde de dados de treino: {}.".format(len(dfdados_train)))
    dfdados_test = pd.read_csv(ARQUIVO_TESTE, sep=';')
    logging.info("Qtde de dados de teste: {}.".format(len(dfdados_test)))

    # Remove os documentos muito grandes    
    # Os dados de originais e permutados foram colocados em uma mesma coluna chamada "documento"
    # , o que diferencia é a classe 0 - Original e 1 - Permutado.
    dfdados_train = descartandoDocumentosGrandesClassificacaoFold(model_args, tokenizer, dfdados_train)
    dfdados_test = descartandoDocumentosGrandesClassificacaoFold(model_args, tokenizer, dfdados_test)
    
    return dfdados_train, dfdados_test        
