# Import das bibliotecas.
import logging  # Biblioteca de logging
import requests # Biblioteca de download
from tqdm.notebook import tqdm as tqdm_notebook # Biblioteca para barra de progresso

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *

# ============================  
def downloadArquivo(url_arquivo, nome_arquivo_destino):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.
    
    Parâmetros:
    `url_arquivo` - URL do arquivos a ser feito download.      
    `nome_arquivo_destino` - Nome do arquivo a ser salvo.      
    '''
    
    # Realiza o download de um arquivo em uma url
    data = requests.get(url_arquivo, stream=True)
    
    if data.status_code != 200:
        loggin.info("Exceção ao tentar realizar download {}. Response {}".format(url, data.status_code), file=sys.stderr)
        data.raise_for_status()
        return

    # Diretório temporário    
    caminho_download = nome_arquivo_destino + "_part"
    
    with open(caminho_download, "wb") as arquivo_binario:        
        tamanho_conteudo = data.headers.get('Content-Length')        
        total = int(tamanho_conteudo) if tamanho_conteudo is not None else None        
        progress = tqdm_notebook(unit="B", total=total, unit_scale=True)        
        for chunk in data.iter_content(chunk_size=1024):        
            if chunk: 
                progress.update(len(chunk))
                arquivo_binario.write(chunk)

    os.rename(caminho_download, nome_arquivo_destino)
    progress.close()

# ============================  
def downloadArquivo1(url_arquivo, nome_arquivo_destino):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.
    
    Parâmetros:
    `url_arquivo` - URL do arquivos a ser feito download.      
    `nome_arquivo_destino` - Nome do arquivo a ser salvo.      
    '''
    
    # Realiza o download de um arquivo em uma url
    data = requests.get(url_arquivo)
    
    # Salva em um arquivo
    arquivo = open(nome_arquivo_destino, 'wb')
    arquivo.write(data.content)

# ============================      
def carregar(nome_arquivo):
    '''
    Carrega um arquivo texto e retorna as linhas como um único parágrafo(texto).
    
    Parâmetros:
    `nome_arquivo` - Nome do arquivo a ser carregado.           
    '''
        
    # Abre o arquivo
    arquivo = open(nome_arquivo, 'r')
    
    paragrafo = ''
    for linha in arquivo:
        linha = linha.splitlines()
        linha = ' '.join(linha)
        # Remove as tags existentes no final das linhas
        linha = remove_tags(linha)
        if linha != '':
            paragrafo = paragrafo + linha.strip() + ' '
     
    # Fecha o arquivo
    arquivo.close()
    
    # Remove os espaços em branco antes e depois do parágrafo
    return paragrafo.strip()

# ============================  
def carregarLista(nome_arquivo):
    '''
    Carrega um arquivo texto e retorna as linhas como uma lista de sentenças(texto).
    
    Parâmetros:
    `nome_arquivo` - Nome do arquivo a ser carregado.           
    '''

    # Abre o arquivo
    arquivo = open(nome_arquivo, 'r')
    
    sentencas = []
    for linha in arquivo:        
        linha = linha.splitlines()
        linha = ' '.join(linha)
        linha = remove_tags(linha)
        if linha != '':
            sentencas.append(linha.strip())
            
    # Fecha o arquivo
    arquivo.close()
    
    return sentencas    

# ============================      
def salvar(nome_arquivo, texto):                       
    '''
    Salva um texto em um arquivo.
     
    Parâmetros:
    `nome_arquivo` - Nome do arquivo a ser salvo.     
    `texto` - Texto a ser salvo.     
     '''

    arquivo = open(nome_arquivo, 'w')
    arquivo.write(str(texto))
    arquivo.close()     
