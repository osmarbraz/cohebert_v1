# Import das bibliotecas.
import logging  # Biblioteca de logging
from enum import Enum # Biblioteca de Enum

class EstrategiasPooling(Enum):
    MEAN = 0 # Média
    MAX = 1 # Máximo
