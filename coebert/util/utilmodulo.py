# Import das bibliotecas.
import re # Biblioteca para expressão regular
import unicodedata # Biblioteca para tratar codificação de caracteres

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
    
def remove_tags(documento):
     '''
     Remove tags de um documento(texto)
     '''
     import re

     documentoLimpo = re.compile('<.*?>')
     return re.sub(documentoLimpo, '', documento)
 

