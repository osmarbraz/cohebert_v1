# Import das bibliotecas.
import requests # Biblioteca para download

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *

def downloadArquivo(url_arquivo, nome_arquivo_destino):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.    
    '''
    # Realiza o download de um arquivo em uma url
    data = requests.get(url_arquivo)
    # Salva em um arquivo
    arquivo = open(nome_arquivo_destino, 'wb')
    arquivo.write(data.content)
    
def carregar(nomeArquivo):
     '''
     Carrega um arquivo texto e retorna as linhas como um único parágrafo(texto)
     '''
     # Linha anterior    
     arquivo = open(nomeArquivo, 'r')
    
     paragrafo = ''
     for linha in arquivo:
         linha = linha.splitlines()
         linha = ' '.join(linha)
         # Remove as tags existentes no final das linhas
         linha = remove_tags(linha)
         if linha != '':
             paragrafo = paragrafo + linha.strip() + ' '
         
     arquivo.close()
     # Remove os espaços em branco antes e depois do parágrafo
     return paragrafo.strip()

def carregarLista(nomeArquivo):
     '''
     Carrega um arquivo texto e retorna as linhas como uma lista de sentenças(texto)
     '''

     # Linha anterior    
     arquivo = open(nomeArquivo, 'r')
     sentencas = []
     for linha in arquivo:        
         linha = linha.splitlines()
         linha = ' '.join(linha)
         linha = remove_tags(linha)
         if linha != '':
            sentencas.append(linha.strip())
     arquivo.close()
     return sentencas    

def salvar(nomeArquivo, texto):                       
     '''
     Salva um texto em um arquivo
     '''

     arquivo = open(nomeArquivo, 'w')
     arquivo.write(str(texto))
     arquivo.close()     
