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
def downloadCSTNewsICMC():  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews do site do ICMC.
    É utilizando um arquivo de entrada compactado. Este arquivo compactado possui diversos arquivos contendo os experimentos realizados. 
    Iremos utilizar os arquivos do experimento ''Modelo de Relaces Discursivas.zip'. 
    Este arquivo descompactado possui dois diretórios de interesse, uma chamada 'Sumarios_Humanos' e outra 'Sumrios_Humanos_Permutados'. 
    '''
  
    # Nome do arquivo a ser recuperado
    NOME_ARQUIVO = "Summarycoherencemodels.zip"

    if os.path.isfile(NOME_ARQUIVO):
        os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do ICMC
    URL_ARQUIVO = "https://sites.icmc.usp.br/taspardo/Summary coherence models.zip"

    # Realiza o download do arquivo dos experimentos    
    downloadArquivo(URL_ARQUIVO, NOME_ARQUIVO)
        
    # Descompacta o arquivo dos experimentos             
    with zipfile.ZipFile(NOME_ARQUIVO, 'r') as arquivoCompactado:
        # Recupera a lista dos nomes dos arquivos dentro do arquivo compactado
        zipInfo = arquivoCompactado.infolist()
        # Percorre a lista dos nomes arquivos compactados
        for arquivo in zipInfo: 
            # Limpa o nome do arquivo
            arquivo.filename = removeAcentos(arquivo.filename)        
            arquivoCompactado.extract(arquivo)
    
    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = "Modelo de Relaces Discursivas.zip"
        
    # Apaga o diretório 'Modelo de Relações Discursivas' e seus arquivos
    if os.path.exists("Modelo de Relacoaes Discursivas"):
        # Apaga o diretório e os arquivos existentes                    
        shutil.rmtree("Modelo de Relacoaes Discursivas")

    # Descompacta o arquivo do experimento               
    with zipfile.ZipFile(NOME_ARQUIVO_EXPERIMENTO, 'r') as arquivoCompactado:
        # Recupera a lista dos nomes dos arquivos dentro do arquivo compactado
        zipInfo = arquivoCompactado.infolist()
        # Percorre a lista dos nomes arquivos compactados
        for arquivo in zipInfo: 
            # Limpa o nome do arquivo
            arquivo.filename = removeAcentos(arquivo.filename)        
            arquivoCompactado.extract(arquivo)

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO_EXPERIMENTO)  

# ============================    
def downloadCSTNewsOnDrive():  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews de um diretório compartilhado do OneDrive.
    '''
  
    # Nome do arquivo a ser recuperado
    NOME_ARQUIVO = "Summarycoherencemodels.zip"

    if os.path.isfile(NOME_ARQUIVO):
        os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do OneDrive
    URL_ARQUIVO = "https://udesc-my.sharepoint.com/:u:/g/personal/91269423991_udesc_br/EQfOLQ6Vg_1Hs4JSwg0aO4wBnxY2ym8tua1XIQB00kczOg?e=hBAqpE&download=1"

    # Realiza o download do arquivo dos experimentos    
    downloadArquivo(URL_ARQUIVO, NOME_ARQUIVO)

    # Descompacta o arquivo dos experimentos             
    with zipfile.ZipFile(NOME_ARQUIVO, 'r') as arquivoCompactado:
        # Recupera a lista dos nomes dos arquivos dentro do arquivo compactado
        zipInfo = arquivoCompactado.infolist()
        # Percorre a lista dos nomes arquivos compactados
        for arquivo in zipInfo: 
            # Limpa o nome do arquivo
            arquivo.filename = removeAcentos(arquivo.filename)        
            arquivoCompactado.extract(arquivo)
    
    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = "Modelo de Relaces Discursivas.zip"
        
    # Apaga o diretório 'Modelo de Relações Discursivas' e seus arquivos
    if os.path.exists("Modelo de Relacoaes Discursivas"):
        # Apaga o diretório e os arquivos existentes                     
        shutil.rmtree("Modelo de Relacoaes Discursivas")

    # Descompacta o arquivo do experimento               
    with zipfile.ZipFile(NOME_ARQUIVO_EXPERIMENTO, 'r') as arquivoCompactado:
        # Recupera a lista dos nomes dos arquivos dentro do arquivo compactado
        zipInfo = arquivoCompactado.infolist()
        # Percorre a lista dos nomes arquivos compactados
        for arquivo in zipInfo: 
            # Limpa o nome do arquivo
            arquivo.filename = removeAcentos(arquivo.filename)        
            arquivoCompactado.extract(arquivo)

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO_EXPERIMENTO)  

# ============================
def carregaArquivosOriginaisCSTNews():  
    '''    
    Carrega os arquivos originais dos arquivos do CSTNews.
    O diretório 'Sumarios_Humanos' contêm os arquivos dos documentos originais onde cada linha representa uma sentença do documento. 
    O nome de cada arquivo original é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_' e um número que identifica o sumário. 
    Os documentos permutados estão no diretório 'Sumarios_Humanos_Permutados' e cada linha representa uma sentença do documento. 
    O nome de cada arquivo de documento permutado é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_', um número que identifica o sumário, o literal 'Perm_' e um número que indica a permutação.
    '''
    
    # Lista dos documentos originais
    lista_documentos_originais = []

    # Lista dos arquivos do diretório
    arquivos = os.listdir("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/")

    # Apaga arquivos desnecessários da lista
    if '.DS_Store' in arquivos:
        arquivos.remove('.DS_Store')

    # Percorre a lista de arquivos
    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]
        
        # Carrega o documento
        documento = carregar("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/" + arquivos[i])
        
        # Carrega as sentenças do documento em uma lista        
        sentencas = carregarLista("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/" + arquivos[i])
        
        # Adiciona o documento e as sentenças a lista
        lista_documentos_originais.append([arquivos[i], sentencas, documento])
    
    logging.info("Carregamento de documento originais concluído: {}.".format(len(lista_documentos_originais)))

    return lista_documentos_originais

# ============================
def carregaArquivosPermutadosCSTNews():  
    '''    
    Carrega os arquivos permutados dos arquivos do CSTNews.
    O diretório 'Sumarios_Humanos' contêm os arquivos dos documentos originais onde cada linha representa uma sentença do documento. 
    O nome de cada arquivo original é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_' e um número que identifica o sumário. 
    Os documentos permutados estão no diretório 'Sumarios_Humanos_Permutados' e cada linha representa uma sentença do documento. 
    O nome de cada arquivo de documento permutado é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_', um número que identifica o sumário, o literal 'Perm_' e um número que indica a permutação.
    '''
  
    # Lista dos documentos permutados
    lista_documentos_permutados = []

    # Lista dos arquivos do diretório
    arquivos = os.listdir("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos_Permutados/") #Entrada (Input) - diretório de sumários humanos e permutados

    # Apaga arquivos desnecessários da lista
    if '.DS_Store' in arquivos:
        arquivos.remove('.DS_Store')
    
    # Percorre a lista de arquivos
    for i in range(len(arquivos)):
        
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        # Carrega o documento
        documento = carregar("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos_Permutados/" + arquivos[i])
        
        # Carrega as sentenças do documento em uma lista
        sentencas = carregarLista("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos_Permutados/" + arquivos[i])

        # Adiciona o documento e as sentenças a lista
        lista_documentos_permutados.append([arquivos[i], sentencas, documento])
    
    logging.info("Carregamento de documento permutados concluído: {}.".format(len(lista_documentos_permutados)))
    
    return lista_documentos_permutados

# ============================
def carregaParesDocumentosCSTNews():
    '''    
    Carrega os arquivos e gera os pares de documentos em uma lista.
    O diretório 'Sumarios_Humanos' contêm os arquivos dos documentos originais onde cada linha representa uma sentença do documento. 
    O nome de cada arquivo original é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_' e um número que identifica o sumário. 
    Os documentos permutados estão no diretório 'Sumarios_Humanos_Permutados' e cada linha representa uma sentença do documento. 
    O nome de cada arquivo de documento permutado é formado um caracter 'C_', um número que identifica o conteúdo, o literal 'Extrato_', um número que identifica o sumário, o literal 'Perm_' e um número que indica a permutação.
    '''
    
    # Lista dos documentos originais e permutados 
    lista_documentos = []

    # Lista dos arquivos do diretório
    arquivosOriginais = os.listdir("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/") #Entrada (Input) - diretório de sumários humanos e permutados
    
    # Apaga arquivos desnecessários da lista
    if '.DS_Store' in arquivosOriginais:
        arquivosOriginais.remove('.DS_Store')

    # Percorre a lista de arquivos
    for i in range(len(arquivosOriginais)):

        # Recupera a posição do ponto no nome do arquivo.
        ponto = arquivosOriginais[i].find('.')
        
        # Recupera o nome do arquivo até a posição do ponto.
        arquivoOriginal = arquivosOriginais[i][:ponto]

        # Carrega o documento original.
        # Carrega como parágrafo
        documentoOriginal = carregar("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/" + arquivosOriginais[i])
        
        # Carrega uma lista das sentenças
        sentencasOriginais = carregarLista("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos/" + arquivosOriginais[i])

        # Percorre as 20 permutações.
        for j in range(20):
            # Recupera o nome do arquivo permutado.
            arquivoPermutado = arquivoOriginal + "_Perm_" + str(j) + ".txt"

            # Carrega o arquivo permutado.
            documentoPermutado = carregar("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos_Permutados/" + arquivoPermutado)
            sentencasPermutadas = carregarLista("/content/Modelo de Relacoaes Discursivas/Sumarios_Humanos_Permutados/" + arquivoPermutado)

            # Adiciona o par original e sua versão permutada.
            lista_documentos.append([arquivosOriginais[i], sentencasOriginais, documentoOriginal, arquivoPermutado, sentencasPermutadas, documentoPermutado])
        
    logging.info("Geração de pares de documentos concluído: {}.".format(len(lista_documentos)))
    
    return lista_documentos

# ============================
def downloadConjuntoDeDados(ORIGEM): 
    '''    
    Verifica de onde será realizado o download dos arquivos de dados.
    
    Parâmetros:        
    `ORIGEM` - Se a variável for setada indica de onde fazer o download.     
    '''
  
    if ORIGEM:
        logging.info("Realizando o download do CSTNews do site do ICMC.")
        downloadCSTNewsICMC()
    else:
        logging.info("Realizando o download do CSTNews do meu OneDrive.")
        downloadCSTNewsOnDrive()

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
def descartandoDocumentosGrandesMedida(model_args, tokenizer, dfdados):
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

        # Remove colunas desnecessárias
        # Para calcular o número de registros removidos não pode existir listas no dataframe
        dfdadosSemLista = dfdadosretorno.drop(columns=['sentencasOriginais', 'sentencasPermutadas'])
        dfdados512SemLista = dfdadosAnterior.drop(columns=['sentencasOriginais', 'sentencasPermutadas'])

        # Registros removidos
        df = dfdados512SemLista.merge(dfdadosSemLista, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        logging.info("Quantidade de registros removidos: {}.".format(len(df)))
        
    else:
        logging.info("Tokenizador não definido.")        
    
    return dfdadosretorno  

# ============================
def getListasDocumentosMedidas(ORIGEM):  
    '''    
    Carrega os arquivos de documentos originais e permutados do CSTNews e retorna suas listas preenchidas.
    
    Parâmetros:        
    `ORIGEM` - Se a variável for setada indica de onde fazer o download. 
    
    Retorno:
    `lista_documentos_originais` - Lista com os documentos originais.
    `lista_documentos_permutados` - Lista com os documentos permutados.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados(ORIGEM)
    
    # Carrega os documentos permutados dos arquivos
    lista_documentos_originais = carregaArquivosOriginaisCSTNews()
    
    # Carrega os documentos permutados dos arquivos
    lista_documentos_permutados = carregaArquivosPermutadosCSTNews()
            
    return lista_documentos_originais, lista_documentos_permutados

# ============================
def getConjuntoDeDadosMedida(model_args, tokenizer, ORIGEM):  
    '''    
    Carrega os dados do CSTNews para o cálculo de medida e retorna um dataframe.
    
    Parâmetros:        
    `model_args` - Objeto com os argumentos do modelo.            
    `tokenizer` - Tokenizador BERT.
    `ORIGEM` - Se a variável for setada indica de onde fazer o download.       
        
    Retorno:
    `dfdados` - Um dataframe com os dados carregados.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados(ORIGEM)
    
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosCSTNews()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosGrandesMedida(model_args, tokenizer, dfdados)
    
    return dfdados       
