# Import das bibliotecas.
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
import pandas as pd # Biblioteca pandas

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

# ============================
def downloadOnlineEducGoogleDrive():
    '''    
    Download dos arquivos do conjunto de dados do OnlineEduc 1.0 do google drive.
    '''
    
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

# ============================    
def carregaArquivosOriginaisOnlineEduc():    
    '''    
    Carrega os arquivos originais dos arquivos do OnlineEduc 1.0.
    '''
  
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
    
    print ('Carregamento de documento originais concluído: ', len(lista_documentos_originais))    

    return lista_documentos_originais

# ============================   
def carregaArquivosPermutadosOnlineEduc():
    '''    
    Carrega os arquivos permutados dos arquivos do OnlineEduc 1.0.
    '''

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
    
    print ('Carregamento de documento permutados concluído: ', len(lista_documentos_permutados))    
    
    return lista_documentos_permutados 

# ============================  
def carregaParesDocumentosOnlineEduc():
    '''    
    Carrega os arquivos e gera os pares de documentos em uma lista.
    '''
  
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

# ============================    
def downloadConjuntoDeDados(): 
    '''    
    Verifica de onde será realizado o download dos arquivos de dados.
    '''
    
    print("Realizando o download do Google Drive.")
    downloadOnlineEducGoogleDrive()
    
# ============================    
def converteListaParesDocumentos(lista_documentos):
    '''    
    Converte a lista de pares de documentos em um dataframe.
    '''

    # Converte a lista em um dataframe.
    dfdados = pd.DataFrame.from_records(lista_documentos, columns=['idOriginal','sentencasOriginais','documentoOriginal','idPermutado','sentencasPermutadas','documentoPermutado'])

    return dfdados
    
# ============================    
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
  
# ============================    
def getConjuntoDeDadosMedida(model_args, tokenizer): 
    '''    
    Carrega os dados do OnlineEduc 1.0 para o cálculo de medida  e retorna um dataframe.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados()
        
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosOnlineEduc()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosMuitoGrandes(dfdados, model_args, tokenizer)
    
    return dfdados  
