# Import das bibliotecas
import time
import datetime

class Util:
  
    '''
    Classe para agrupar funções utilitárias.    
    '''

    # Construtor da classe    
    def __init__(self):
    
    def formataTempo(tempo):
        '''
        Pega a tempo em segundos e retorna uma string hh:mm:ss
        '''
        # Arredonda para o segundo mais próximo.
        tempoArredondado = int(round((tempo)))
    
        # Formata como hh:mm:ss
        return str(datetime.timedelta(seconds=tempoArredondado))
