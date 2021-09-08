# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca para manipular arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

# ============================
def downloadOnlineEducGoogleDrive():
    '''    
    Download dos arquivos do conjunto de dados do OnlineEduc 1.0 do Google Drive.
    Depende de mapeamento no Google Drive.
    '''
    
    # Diretório do Google Drive
    DIRETORIO_GOOGLEDRIVE = "/content/drive"

    # Verifica se o diretório do google drive foi montada
    if os.path.exists(DIRETORIO_GOOGLEDRIVE): 
        
        logging.info("Realizando a cópia do arquivo de dados do OnlineEduc 1.0 do Google Drive.")
    
        # Nome do arquivo
        NOME_ARQUIVO_ORIGINAL = "original.zip"
        NOME_ARQUIVO_PERMUTADO = "permutado.zip"

        # Define o caminho e nome do arquivo de dados
        CAMINHO_ARQUIVO_ORIGINAL = "/content/drive/MyDrive/Colab Notebooks/Data/Moodle/dadosmoodle_documento_pergunta_sentenca_intervalo/" + NOME_ARQUIVO_ORIGINAL
        CAMINHO_ARQUIVO_PERMUTADO = "/content/drive/MyDrive/Colab Notebooks/Data/Moodle/dadosmoodle_documento_pergunta_sentenca_intervalo/" + NOME_ARQUIVO_PERMUTADO

        # Copia o arquivo do modelo para o diretório no Google Drive.
        shutil.copy(CAMINHO_ARQUIVO_ORIGINAL, '.') 
        shutil.copy(CAMINHO_ARQUIVO_PERMUTADO, '.') 

        # Descompacta o arquivo no diretório de descompactação.                
        arquivoZip = zipfile.ZipFile(NOME_ARQUIVO_ORIGINAL, "r")
        arquivoZip.extractall()

        # Descompacta o arquivo no diretório de descompactação.                
        arquivoZip = zipfile.ZipFile(NOME_ARQUIVO_PERMUTADO, "r")
        arquivoZip.extractall()
        
    else:        
        logging.info("Diretório do google drive não foi montado: {}.".format(DIRETORIO_GOOGLEDRIVE))
        

# ============================    
def carregaArquivosOriginaisOnlineEduc():    
    '''    
    Carrega os arquivos originais dos arquivos do OnlineEduc 1.0.
    '''
  
    lista_documentos_originais = []

    arquivos = os.listdir("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/") 

    # Percorre a lista de arquivos do diretório
    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        # Carrega o arquivo de nome x[i] do diretório
        documento = carregar("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/" + arquivos[i])
        sentencas = carregarLista("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/" + arquivos[i])

        lista_documentos_originais.append([nomeArquivo, sentencas, documento])
    
    logging.info("Carregamento de documento originais do OnlineEduc 1.0 concluído: {}.".format(len(lista_documentos_originais)))

    return lista_documentos_originais

# ============================   
def carregaArquivosPermutadosOnlineEduc():
    '''    
    Carrega os arquivos permutados dos arquivos do OnlineEduc 1.0.
    '''

    lista_documentos_permutados = []

    arquivos = os.listdir("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/") #Entrada (Input)

    # Percorre a lista de arquivos do diretório
    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        # Carrega o arquivo de nome x[i] do diretório
        documento = carregar("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/" + arquivos[i])
        sentencas = carregarLista("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/" + arquivos[i])

        # Adiciona a lista o conteúdo do arquivo
        lista_documentos_permutados.append([nomeArquivo, sentencas, documento])
    
    logging.info("Carregamento de documento permutados do OnlineEduc 1.0 concluído: {}.".format(len(lista_documentos_permutados)))
    
    return lista_documentos_permutados 

# ============================  
def carregaParesDocumentosOnlineEduc():
    '''    
    Carrega os arquivos e gera os pares de documentos em uma lista.
    '''
  
    # Lista dos documentos originais e permutados 
    lista_documentos = []

    arquivosOriginais = os.listdir("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/") 

    for i in range(len(arquivosOriginais)):

        # Recupera a posição do ponto no nome do arquivo.
        ponto = arquivosOriginais[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto.
        arquivoOriginal = arquivosOriginais[i][:ponto]

        # Carrega o documento original.
        # Carrega como parágrafo
        documentoOriginal = carregar("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/" + arquivosOriginais[i])
        # Carrega uma lista das sentenças
        sentencasOriginais = carregarLista("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/" + arquivosOriginais[i])

        # Percorre as 20 permutações.
        for j in range(20):
            # Recupera o nome do arquivo permutado.
            arquivoPermutado = arquivoOriginal + '_Perm_' + str(j) + '.txt'

            # Carrega o arquivo permutado.
            documentoPermutado = carregar("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/" + arquivoPermutado)
            sentencasPermutadas = carregarLista("/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/" + arquivoPermutado)

            # Adiciona o par original e sua versão permutada.
            lista_documentos.append([arquivosOriginais[i], sentencasOriginais, documentoOriginal, arquivoPermutado, sentencasPermutadas, documentoPermutado])

    logging.info("Geração de pares concluído: {}.".format(len(lista_documentos)))
    
    return lista_documentos

# ============================    
def downloadConjuntoDeDados(): 
    '''    
    Verifica de onde será realizado o download dos arquivos de dados.
    '''
        
    downloadOnlineEducGoogleDrive()
    
# ============================    
def converteListaParesDocumentos(lista_documentos):
    ''' 
    Converte a lista de pares(lista_documentos) de documentos em um dataframe.
    Atributos do dataframe:
        0. 'idOriginal' - Nome do arquivo original.
        1. 'sentencasOriginais' - Lista das sentenças do documento original.
        2. 'documentoOriginal' - Documento original.
        3. 'idPermutado' - Nome do arquivo permutado.
        4. 'sentencasPermutadas' - Lista das sentenças do documento permtuado.
        5. 'documentoPermutado' - Documento permutado.
        
    Parâmetros:
    `lista_documentos` - Lista de pares de documentos. 
    
    Retorno:
    `dfdados` - Um dataframe com os dados carregados.    
    '''

    # Converte a lista em um dataframe.
    dfdados = pd.DataFrame.from_records(lista_documentos, columns=['idOriginal', 'sentencasOriginais', 'documentoOriginal', 'idPermutado', 'sentencasPermutadas', 'documentoPermutado'])

    return dfdados
    
# ============================    
def descartandoDocumentosGrandes(model_args, tokenizer, dfdados):
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
        dfdados['input_ids'] = dfdados['documento'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

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
def getListasDocumentosMedidas(ORIGEM):  
    '''    
    Carrega os arquivos de documentos originais e permutados do OnlineEduc 1.0 e retorna suas listas preenchidas.
    
    Parâmetros:        
    `ORIGEM` - Se a variável for setada indica de onde fazer o download.
    
    Retorno:
    `lista_documentos_originais` - Lista com os documentos originais.
    `lista_documentos_permutados` - Lista com os documentos permutados.
    '''
    
    # # Realiza o download do conjunto de dados
    downloadConjuntoDeDados()
        
    # Carrega os documentos permutados dos arquivos
    lista_documentos_originais = carregaArquivosOriginaisOnlineEduc()
    
    # Carrega os documentos permutados dos arquivos
    lista_documentos_permutados = carregaArquivosPermutadosOnlineEduc()
            
    return lista_documentos_originais, lista_documentos_permutados

# ============================    
def getConjuntoDeDadosMedida(model_args, tokenizer): 
    '''    
    Carrega os dados do OnlineEduc 1.0 para o cálculo de medida  e retorna um dataframe.  
    
    Parâmetros:        
    `model_args` - Objeto com os argumentos do modelo.
    `tokenizer` - Tokenizador BERT.
    
    Retorno:
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
    
    return dfdados  
