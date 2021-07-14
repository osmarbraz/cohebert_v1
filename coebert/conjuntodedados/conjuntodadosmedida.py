
# Import das bibliotecas.
import wget # Biblioteca de download
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas

def downloadCSTNewsICMC():
  
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'Summarycoherencemodels.zip'

    if os.path.isfile(NOME_ARQUIVO)
       os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do ICMC.
    URL_ARQUIVO = 'https://sites.icmc.usp.br/taspardo/Summary%20coherence%20models.zip'  

    # Realiza o download do arquivo do modelo
    wget.download(URL_ARQUIVO, out=NOME_ARQUIVO)

    # Descompacta o arquivo dos experimentos             
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO,"r")
    arquivoZip.extractall()

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = 'Modelo de RelaЗфes Discursivas.zip'

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
  
    # Nome do arquivo a ser criado.
    NOME_ARQUIVO = 'Summarycoherencemodels.zip'

    if os.path.isfile(NOME_ARQUIVO)
       os.remove(NOME_ARQUIVO)

    # Realiza o download do arquivo do OneDrive.
    URL_ARQUIVO = 'https://udesc-my.sharepoint.com/:u:/g/personal/91269423991_udesc_br/EQfOLQ6Vg_1Hs4JSwg0aO4wBnxY2ym8tua1XIQB00kczOg?e=hBAqpE&download=1'

    #!wget -O '$NOME_ARQUIVO' --no-check-certificate  

    # Realiza o download do arquivo do modelo
    wget.download(URL_ARQUIVO, out=NOME_ARQUIVO)

    # Descompacta o arquivo dos experimentos             
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO,"r")
    arquivoZip.extractall()

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO)

    # Especifica o nome do arquivo do experimento
    NOME_ARQUIVO_EXPERIMENTO = 'Modelo de RelaЗфes Discursivas.zip'

    # Apaga o diretório 'Modelo de Relações Discursivas' e seus arquivos
    if os.path.exists('Modelo de Relações Discursivas'):
        # Apaga a pasta e os arquivos existentes                     
        shutil.rmtree('Modelo de Relações Discursivas')

    # Descompacta o arquivo o experimento
    arquivoZip = zipfile.ZipFile(NOME_ARQUIVO_EXPERIMENTO,"r")
    arquivoZip.extractall()

    # Apaga o arquivo compactado
    os.remove(NOME_ARQUIVO_EXPERIMENTO)  

def downloadOnlineEducGooglDrive():
    # Nome do arquivo
    NOMEARQUIVOORIGINAL = 'original.zip'
    NOMEARQUIVOPERMUTADO = 'permutado.zip'
    
    # Define o caminho e nome do arquivo de dados
    CAMINHOARQUIVOORIGINAL = '/content/drive/MyDrive/Colab Notebooks/Data/Moodle/dadosmoodle_documento_pergunta_sentenca_intervalo/' + NOMEARQUIVOORIGINAL
    CAMINHOARQUIVOPERMUTADO = '/content/drive/MyDrive/Colab Notebooks/Data/Moodle/dadosmoodle_documento_pergunta_sentenca_intervalo/' + NOMEARQUIVOPERMUTADO
    
    # Copia o arquivo do modelo para o diretório no Google Drive.
    shutil.copy(CAMINHOARQUIVOORIGINAL, '.') 
    shutil.copy(CAMINHOARQUIVOPERMUTADO, '.') 

    # Descompacta o arquivo na pasta de descompactação.                
    arquivoZip = zipfile.ZipFile(NOMEARQUIVOORIGINAL,"r")
    arquivoZip.extractall()
    
    # Descompacta o arquivo na pasta de descompactação.                
    arquivoZip = zipfile.ZipFile(NOMEARQUIVOPERMUTADO,"r")
    arquivoZip.extractall()
    
def carregaArquivosOriginaisOnlineEduc():    
  
    lista_documentos_originais = []

    arquivos = os.listdir('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/') 

    # Percorre a lista de arquivos do diretório
    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        ponto = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:ponto]

        # Carrega o arquivo de nome x[i] do diretório
        documento = carregar('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/' + arquivos[i])
        sentencas = carregarLista('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/' + arquivos[i])

        lista_documentos_originais.append([nomeArquivo, sentencas, documento])

    print ('TERMINADO ORIGINAL: ', len(lista_documentos_originais))
    retirm lista_documentos_originais
    

def carregaArquivosPermutadosOnlineEduc():    

    lista_documentos_permutados = []

    arquivos = os.listdir('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/') #Entrada (Input)

    # Percorre a lista de arquivos do diretório
    for i in range(len(arquivos)):
        # Recupera a posição do ponto no nome do arquivo
        w = arquivos[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto
        nomeArquivo = arquivos[i][:w]

        # Carrega o arquivo de nome x[i] do diretório
        documento = carregar('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/' + arquivos[i])
        sentencas = carregarLista('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/' + arquivos[i])

        # Adiciona a lista o conteúdo do arquivo
        lista_documentos_permutados.append([nomeArquivo, sentencas, documento])

    print ('TERMINADO PERMUTADO: ', len(lista_documentos_permutados))
    return lista_documentos_permutados 
  
def carregaParesDocumentosOnlineEduc():  
  
    # Lista dos documentos originais e permutados 
    lista_documentos = []

    arquivosOriginais = os.listdir('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/') #Entrada (Input)

    for i in range(len(arquivosOriginais)):

        # Recupera a posição do ponto no nome do arquivo.
        ponto = arquivosOriginais[i].find('.')
        # Recupera o nome do arquivo até a posição do ponto.
        arquivoOriginal = arquivosOriginais[i][:ponto]

        # Carrega o documento original.
        # Carrega como parágrafo
        documentoOriginal = carregar('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/' + arquivosOriginais[i])
        # Carrega uma lista das sentenças
        sentencasOriginais = carregarLista('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/original/' + arquivosOriginais[i])

        # Percorre as 20 permutações.
        for j in range(20):
            # Recupera o nome do arquivo permutado.
            arquivoPermutado = arquivoOriginal + '_Perm_'+str(j) + '.txt'

            # Carrega o arquivo permutado.
            documentoPermutado = carregar('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/' + arquivoPermutado)
            sentencasPermutadas = carregarLista('/content/dadosmoodle_documento_pergunta_sentenca_intervalo/permutado/' + arquivoPermutado)

            # Adiciona o par original e sua versão permutada.
            lista_documentos.append([arquivosOriginais[i], sentencasOriginais, documentoOriginal, arquivoPermutado, sentencasPermutadas, documentoPermutado])

    print ('TERMINADO GERAÇÃO PARES:', len(lista_documentos))
    
    return lista_documentos
    
def carregaArquivosOriginaisCSTNEWS():
  
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


def carregaArquivosPermutadosCSTNEWS():
  
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

def carregaParesDocumentosCSTNEWS():
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
    
def downloadConjuntoDeDados(TIPO = 0, ORIGEM='ICMC'): #0 - CSTNews ou 1 - OnlineEduc 1.0
  
    if TIPO == 0:
        # Pode ser realizado o download de dois lugares
        if ORIGEM ==  'ICMC':
            downloadCSTNewsICMC()
        else:
            downloadCSTNewsOnDrive()
    else:
        downloadOnLineEduc()
    
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
  
    
def getConjuntoDeDadosMedida(TIPO = 0, model_args, ORIGEM='ICMC'): #0 - CSTNews ou 1 - OnlineEduc 1.0    
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados(TIPO,ORIGEM='ICMC', tokenizer)
    
    # Verifica o tipo do conjunto de dados se é 0 - CSTNews ou 1 - OnlineEduc 1.0    
    if TIPO == 0:
        # Carrega os pares de documentos dos arquivos
        lista_documentos = carregaParesDocumentosCSTNEWS()
        
        # Converte em um dataframe
        dfdados = converteListaParesDocumentosCSTNEWS(lista_documentos)
        
        # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
        dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    else:
      
       # Carrega os pares de documentos dos arquivos
        lista_documentos = carregaParesDocumentosOnlineEduc()
        
        # Converte em um dataframe
        dfdados = converteListaParesDocumentos(lista_documentos)
        
        # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
        dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    
    return dfdados    
    
