# Import das bibliotecas.
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

def similaridadeCoseno(documento1, documento2):
    '''
    Similaridade do cosseno dos embeddgins das senten�as.
    '''
    similaridade = 1 - cosine(documento1, documento2)
    return similaridade

def distanciaEuclidiana(sentenca1, sentenca2):
    '''
    Dist�ncia euclidiana entre os embeddings das senten�as.
    Possui outros nomes como dist�ncia L2 ou norma L2.
    '''

    distancia = euclidean(sentenca1, sentenca2)
    return distancia


def distanciaManhattan(sentenca1, sentenca2):
    '''
    Dist�ncia Manhattan entre os embeddings das senten�as.
    Possui outros nomes como dist�ncia Cityblock, dist�ncia L1, norma L1 e m�trica do t�xi.
    '''
    distancia = cityblock(sentenca1, sentenca2)

    return distancia