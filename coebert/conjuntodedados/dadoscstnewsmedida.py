# Import das bibliotecas.
import requests # Biblioteca para download
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas
import re # Biblioteca para expressão regular
import unicodedata # Biblioteca para tratar codificação de caracteres

def removeAcentos(texto):   
    try:
        text = unicode(texto, 'utf-8')
    except (TypeError, NameError): 
        pass
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore')
    texto = texto.decode("utf-8")
    return str(texto)

def limpaTexto(texto):    
    texto = removeAcentos(texto.lower())
    texto = re.sub('[ ]+', '_', texto)
    texto = re.sub('[^.0-9a-zA-Z_-]', '', texto)
    return texto

def downloadCSTNewsICMC():  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews do site do ICMC.
    '''
  
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'Summarycoherencemodels.zip'

    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do ICMC.
    URL_ARQUIVO = 'https://sites.icmc.usp.br/taspardo/Summary coherence models.zip'  

    # Realiza o download do arquivo do modelo
    data = requests.get(URL_ARQUIVO)
    arquivo = open(NOME_ARQUIVO, 'wb')
    arquivo.write(data.content)

    # Descompacta o arquivo dos experimentos             
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO,"r")
    arquivoZip.extractall()

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = 'Modelo de RelaçΣes Discursivas.zip'

    # Apaga o diretório 'Modelo de Relações Discursivas' e seus arquivos
    if os.path.exists('Modelo de Relações Discursivas'):
        # Apaga a pasta e os arquivos existentes                     
        shutil.rmtree('Modelo de Relações Discursivas')

    # Descompacta o arquivo o experimento
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO_EXPERIMENTO,"r")
    arquivoZip.extractall()

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO_EXPERIMENTO)
  
def downloadCSTNewsOnDrive():
  
    '''    
    Download dos arquivos do conjunto de dados do CSTNews de uma pasta compartilhada do One Drive.
    '''
  
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'Summarycoherencemodels.zip'

    if os.path.isfile(NOME_ARQUIVO):
       os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do OneDrive.
    URL_ARQUIVO = 'https://udesc-my.sharepoint.com/:u:/g/personal/91269423991_udesc_br/EQfOLQ6Vg_1Hs4JSwg0aO4wBnxY2ym8tua1XIQB00kczOg?e=hBAqpE&download=1'

    # Realiza o download do arquivo do modelo
    data = requests.get(URL_ARQUIVO)
    arquivo = open(NOME_ARQUIVO, 'wb')
    arquivo.write(data.content)

    # Descompacta o arquivo dos experimentos             
    with zipfile.ZipFile(NOME_ARQUIVO, 'r') as f:
      zipInfo = f.infolist()
      for arquivo in zipInfo: 
        # Limpa o nome do arquivo
        arquivo.filename = limpaTexto(arquivo.filename)        
        f.extract(arquivo)
    
    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = 'modelo_de_relaces_discursivas.zip'

    # Apaga o diretório 'Modelo de Relações Discursivas' e seus arquivos
    if os.path.exists('Modelo de Relações Discursivas'):
        # Apaga a pasta e os arquivos existentes                     
        shutil.rmtree('Modelo de Relações Discursivas')

    # Descompacta o arquivo do experimento               
    with zipfile.ZipFile(NOME_ARQUIVO_EXPERIMENTO, 'r') as f:
      zipInfo = f.infolist()
      for arquivo in zipInfo: 
        # Limpa o nome do arquivo
        arquivo.filename = limpaTexto(arquivo.filename)        
        f.extract(arquivo)

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO_EXPERIMENTO)  

    
def carregaArquivosOriginaisCSTNews():  
    '''    
    Carrega os arquivos originais dos arquivos do CSTNews.
    '''
  
    lista_documentos_originais = []

    arquivos = os.listdir('/content/Modelo de Relações Discursivas/Sumarios_Humanos/') #Entrada (Input) - diretório de sumários humanos e permutados

    if '.DS_Store' in arquivos:
      arquivos.remove('.DS_Store')

    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        documento = carregar('/content/Modelo de Relações Discursivas/Sumarios_Humanos/'+arquivos[i])
        sentencas = carregarLista('/content/Modelo de Relações Discursivas/Sumarios_Humanos/'+arquivos[i])

        lista_documentos_originais.append([arquivos[i], sentencas, documento])

    print ('TERMINADO ORIGINAIS: ', len(lista_documentos_originais))    

    return lista_documentos_originais

def carregaArquivosPermutadosCSTNews():  
    '''    
    Carrega os arquivos permutados dos arquivos do CSTNews.
    '''
  
    lista_documentos_permutados = []

    arquivos = os.listdir('/content/Modelo de Relações Discursivas/Sumarios_Humanos_Permutados/') #Entrada (Input) - diret�rio de sum�rios humanos e permutados

    if '.DS_Store' in arquivos:
        arquivos.remove('.DS_Store')

    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        documento = carregar('/content/Modelo de Relações Discursivas/Sumarios_Humanos_Permutados/'+arquivos[i])
        sentencas = carregarLista('/content/Modelo de Relações Discursivas/Sumarios_Humanos_Permutados/'+arquivos[i])

        lista_documentos_permutados.append([arquivos[i], sentencas, documento])

    print ('TERMINADO PERMUTADOS: ', len(lista_documentos_permutados)) 

    return lista_documentos_permutados

def carregaParesDocumentosCSTNews():
    '''    
    Carrega os arquivos e gera os pares de documentos em uma lista.
    '''
    
    # Lista dos documentos originais e permutados 
    lista_documentos = []

    arquivosOriginais = os.listdir('/content/Modelo de Relações Discursivas/Sumarios_Humanos/') #Entrada (Input) - diretório de sumários humanos e permutados
    
    if '.DS_Store' in arquivosOriginais:
        arquivosOriginais.remove('.DS_Store')

    for i in range(len(arquivosOriginais)):

        # Recupera a posição do ponto no nome do arquivo.
        ponto = arquivosOriginais[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto.
        arquivoOriginal = arquivosOriginais[i][:ponto]

        # Carrega o documento original.
        # Carrega como parágrafo
        documentoOriginal = carregar('/content/Modelo de Relações Discursivas/Sumarios_Humanos/'+arquivosOriginais[i])
        # Carrega uma lista das sentenças
        sentencasOriginais = carregarLista('/content/Modelo de Relações Discursivas/Sumarios_Humanos/'+arquivosOriginais[i])

        # Percorre as 20 permutações.
        for j in range(20):
            # Recupera o nome do arquivo permutado.
            arquivoPermutado = arquivoOriginal + '_Perm_'+str(j) + '.txt'

            # Carrega o arquivo permutado.
            documentoPermutado = carregar('/content/Modelo de Relações Discursivas/Sumarios_Humanos_Permutados/'+ arquivoPermutado)
            sentencasPermutadas = carregarLista('/content/Modelo de Relações Discursivas/Sumarios_Humanos_Permutados/'+ arquivoPermutado)

            # Adiciona o par original e sua versão permutada.
            lista_documentos.append([arquivosOriginais[i], sentencasOriginais, documentoOriginal, arquivoPermutado, sentencasPermutadas, documentoPermutado])
    
    print ('TERMINADO GERAÇÃO PARES:', len(lista_documentos))
    
    return lista_documentos
    
def downloadConjuntoDeDados(ORIGEM): 
    '''    
    Verifica de onde será realizado o download dos arquivos de dados.
    '''
  
    if ORIGEM:
       downloadCSTNewsICMC()
    else:
       downloadCSTNewsOnDrive()
    
def converteListaParesDocumentos(lista_documentos):
    '''    
    Converte a lista de pares de documentos em um dataframe.
    '''

    # Converte a lista em um dataframe.
    dfdados = pd.DataFrame.from_records(lista_documentos, columns=['idOriginal','sentencasOriginais','documentoOriginal','idPermutado','sentencasPermutadas','documentoPermutado'])

    return dfdados    
    
def descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer):
    '''    
    Remove os documentos que extrapolam 512 tokens.
    '''
  
    # Tokenize a codifica os documentos para o BERT.     
    dfdados['input_ids'] = dfdados['documentoOriginal'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

    # Reduz para o tamanho máximo suportado pelo BERT.
    dfdados_512 = dfdados[dfdados['input_ids'].apply(len)<=model_args.max_seq_len]

    # Remove as colunas desnecessárias.
    dfdadosAnterior = dfdados.drop(columns=['input_ids'])
    dfdadosretorno = dfdados_512.drop(columns=['input_ids'])

    #print('Quantidade de dados anterior: {}'.format(len(dfdadosAnterior)))
    #print('Nova quantidade de dados    : {}'.format(len(dfdadosretorno)))

    # Mostra a quantidade registros removidos
    dfdadosSemLista =  dfdadosretorno.drop(columns=['sentencasOriginais','sentencasPermutadas'])
    dfdados512SemLista =  dfdadosAnterior.drop(columns=['sentencasOriginais','sentencasPermutadas'])

    df = dfdados512SemLista.merge(dfdadosSemLista, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
    #print('Quantidade de registros removidos: {}'.format(len(df)))

    return dfdadosretorno  
  
    
def getConjuntoDeDadosMedida(model_args, ORIGEM, tokenizer):  
    '''    
    Carrega os dados do CSTNews e retorna um dataframe.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados(ORIGEM)
    
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosCSTNEWS()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentosCSTNEWS(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    
    return dfdados       
