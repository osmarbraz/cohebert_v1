def formataTempo(tempo):
     '''
     Pega a tempo em segundos e retorna uma string hh:mm:ss
     '''
     import time
     import datetime
        
     # Arredonda para o segundo mais pr�ximo.
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
     Carrega um arquivo texto e retorna as linhas como um �nico par�grafo(texto)
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
