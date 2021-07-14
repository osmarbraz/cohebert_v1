# Import das bibliotecas.
import re # Biblioteca para expressão regular
import unicodedata # Biblioteca para tratar codificação de caracteres
import requests # Biblioteca para download

def downloadArquivo(url_arquivo, nome_arquivo_destino):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.    
    '''
    # Realiza o download do arquivo dos experimentos
    data = requests.get(url_arquivo)
    arquivo = open(nome_arquivo_destino, 'wb')
    arquivo.write(data.content)

def removeAcentos(texto):   
    '''    
    Remove acentos de um texto.
    '''
    try:
        text = unicode(texto, 'utf-8')
    except (TypeError, NameError): 
        pass
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore')
    texto = texto.decode("utf-8")
    return str(texto)

def limpaTexto(texto):    
    '''    
    Remove acentos e espaços e outros caracteres de um texto.
    '''
    texto = removeAcentos(texto.lower())
    texto = re.sub('[ ]+', '_', texto)
    texto = re.sub('[^.0-9a-zA-Z_-]', '', texto)
    return texto

def formataTempo(tempo):
     '''
     Pega a tempo em segundos e retorna uma string hh:mm:ss
     '''
     import time
     import datetime
        
     # Arredonda para o segundo mais próximo.
     tempoArredondado = int(round((tempo)))
   
     # Formata como hh:mm:ss
     return str(datetime.timedelta(seconds=tempoArredondado))
    
def remove_tags(documento):
     '''
     Remove tags de um documento(texto)
     '''
     import re

     documentoLimpo = re.compile('<.*?>')
     return re.sub(documentoLimpo, '', documento)
  
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

def salvar(nomeArquivo,texto):                       
     '''
     Salva um texto em um arquivo
     '''

     arquivo = open(nomeArquivo, 'w')
     arquivo.write(str(texto))
     arquivo.close() 
