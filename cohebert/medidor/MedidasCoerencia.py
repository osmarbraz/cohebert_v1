# Import das bibliotecas.
import logging  # Biblioteca de logging
from enum import Enum # Biblioteca de Enum

class MedidasCoerencia(Enum):
    COSSENO = 0 # Similaridade do Cosseno
    EUCLIDIANA = 1 # Distância Euclidiana
    MANHATTAN = 2 # Distância de Manhattan
    
