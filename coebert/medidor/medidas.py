# Import das bibliotecas.
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

def similaridadeCoseno(documento1, documento2):
    '''
    Similaridade do cosseno dos embeddgins das sentenças.
    '''
    
    similaridade = 1 - cosine(documento1, documento2)
    return similaridade

def distanciaEuclidiana(sentenca1, sentenca2):
    '''
    Distância euclidiana entre os embeddings das sentenças.
    Possui outros nomes como distância L2 ou norma L2.
    '''
    
    distancia = euclidean(sentenca1, sentenca2)
    return distancia

def distanciaManhattan(sentenca1, sentenca2):
    '''
    Distância Manhattan entre os embeddings das sentenças. 
    Possui outros nomes como distância Cityblock, distância L1, norma L1 e métrica do táxi.
    '''
    
    distancia = cityblock(sentenca1, sentenca2)

    return distancia
