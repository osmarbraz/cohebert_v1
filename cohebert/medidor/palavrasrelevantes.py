# Import das bibliotecas.
import logging  # Biblioteca de logging
from enum import Enum # Biblioteca de Enum

class PalavrasRelevantes(Enum):
    ALL = 0 # Todas as palavras
    CLEAN = 1 # Sem stopwords
    NOUN = 2 # Somente substantivos
