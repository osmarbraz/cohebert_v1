# Import das bibliotecas.
import logging  # Biblioteca de logging
import requests # Biblioteca de download
from tqdm.notebook import tqdm as tqdm_notebook # Biblioteca para barra de progresso
import os # Biblioteca para manipular arquivos

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *

# Diretório do cohebert
DIRETORIO_COHEBERT = 'cohebert_v1'

# ============================  
def verificaDiretorioCoheBERT():
    '''    
    Verifica se existe o diretório cohebert_v1 no diretório corrente.    
    '''
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO_COHEBERT):  
        # Cria o diretório
        os.makedirs(DIRETORIO_COHEBERT)
        logging.info("Diretório Cohebert criado: {}".format(DIRETORIO_COHEBERT))
    else:
        logging.info("Diretório Cohebert já existe: {}".format(DIRETORIO_COHEBERT))

    return DIRETORIO_COHEBERT

# ============================  
def downloadArquivo(url_arquivo, nome_arquivo_destino):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.
    
    Parâmetros:
    `url_arquivo` - URL do arquivo a ser feito download.      
    `nome_arquivo_destino` - Nome do arquivo a ser salvo.      
    '''
    
    # Realiza o download de um arquivo em uma url
    data = requests.get(url_arquivo, stream=True)
    
    # Verifica se o arquivo existe
    if data.status_code != 200:
        loggin.info("Exceção ao tentar realizar download {}. Response {}".format(url, data.status_code), file=sys.stderr)
        data.raise_for_status()
        return

    # Verifica se existe o diretório base
    DIRETORIO_COHEBERT = verificaDiretorioCoheBERT()

    # Arquivo temporário    
    nome_arquivo_temporario = DIRETORIO_COHEBERT + "/" + nome_arquivo_destino + "_part"
    
    logging.info("Download do arquivo: {}.".format(nome_arquivo_destino))
    
    # Baixa o arquivo
    with open(nome_arquivo_temporario, "wb") as arquivo_binario:        
        tamanho_conteudo = data.headers.get('Content-Length')        
        total = int(tamanho_conteudo) if tamanho_conteudo is not None else None
        # Barra de progresso de download
        progresso_bar = tqdm_notebook(unit="B", total=total, unit_scale=True)                
        # Atualiza a barra de progresso
        for chunk in data.iter_content(chunk_size=1024):        
            if chunk:                
                progresso_bar.update(len(chunk))
                arquivo_binario.write(chunk)
    
    # Renomeia o arquivo temporário para o arquivo definitivo
    os.rename(nome_arquivo_temporario, nome_arquivo_destino)
    
    # Fecha a barra de progresso.
    progresso_bar.close()

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
