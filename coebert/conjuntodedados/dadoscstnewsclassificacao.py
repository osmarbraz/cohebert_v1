# Import das bibliotecas.
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas
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
    dfdados_train, dfdados_test = dfdados, test_size=test_qtde, random_state=42, stratify=dfdados[classeStratify])

    print("Conjunto total:", len(dfdados))
    print("  Treino:", len(dfdados_train))
    print("  Teste :", len(dfdados_test))

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

    print("Download do Github")  

    # Diretório dos arquivos de dados
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório
        os.makedirs(DIRETORIO)
        print('Diretório criado: {}'.format(DIRETORIO))
    else:
        print('Diretório já existe: {}'.format(DIRETORIO))
        
    # Download do arquivo de dados  
    
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'CSTNEWS_MD_KFOLD_10.zip'

    # Apaga o arquivo.    
    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)
    
    # Realiza o download do arquivo do OneDrive
    URL_ARQUIVO = 'https://github.com/osmarbraz/coebert/blob/main/conjuntodedados/'+ NOME_ARQUIVO + '?raw=true'

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
    
    print("Copiando do checkout do Github")

    # Diretório dos arquivos de dados.
    DIRETORIO = "/content/validacao_kfold"
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO):  
        # Cria o diretório
        os.makedirs(DIRETORIO)
        print('Diretório criado: {}'.format(DIRETORIO))
    else:
        print('Diretório já existe: {}'.format(DIRETORIO))
        
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = "CSTNEWS_MD_KFOLD_10.zip"
    
    # Diretórios dos arquivos
    DIRETORIO_FONTE_ARQUIVO = "/content/coebert_v1/conjuntodedados/cstnews/" + NOME_ARQUIVO
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
        
        print("Removendo documentos grandes, acima de ", tamanho_maximo, " tokens.")

        # Tokenize a codifica as setenças para o BERT     
        dfdados_train['input_ids'] = dfdados_train['documento'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        dfdados_train = dfdados_train[dfdados_train['input_ids'].apply(len)<tamanho_maximo]

        print('Tamanho do dataset de treino: {}'.format(len(dfdados_train)))

        # Remove as colunas desnecessárias
        dfdados_train = dfdados_train.drop(columns=['input_ids'])

        # Tokenize a codifica as setenças para o BERT.     
        dfdados_test['input_ids'] = dfdados_test['documento'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        # Corta os inputs para o tamanho máximo 512
        dfdados_test = dfdados_test[dfdados_test['input_ids'].apply(len)<tamanho_maximo]

        print('Tamanho do dataset de teste: {}'.format(len(dfdados_test)))

        # Remove as colunas desnecessárias
        dfdados_test = dfdados_test.drop(columns=['input_ids'])
    else:
        print("Tokenizador não definido.")        

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

    print("Carregando arquivo de treino: {}".format(ARQUIVO_TREINO))
    print("Carregando arquivo de teste: {}".format(ARQUIVO_TESTE))

    # Carrega o dataset de treino e teste.
    dfdados_train = pd.read_csv(ARQUIVO_TREINO, sep=';')
    print('Qtde de dados de treino: {}'.format(len(dfdados_train)))
    dfdados_test = pd.read_csv(ARQUIVO_TESTE, sep=';')
    print('Qtde de dados de teste: {}'.format(len(dfdados_test)))

    # Remove os documentos muito grandes
    dfdados_train, dfdados_test = descartandoDocumentosGrandes(tokenizer, model_args, dfdados_train, dfdados_test)
    
    return dfdados_train, dfdados_test        
