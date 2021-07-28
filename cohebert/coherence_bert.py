# Biblioteca cohebert
from bert.bertmodulo import *
from bert.bertarguments import ModeloArgumentosMedida
from experimento.calculomedida import *
from spacynlp.spacymodulo import *

# Definição dos parâmetros do Modelo para os cálculos das Medidas
model_args = ModeloArgumentosMedida(     
    max_seq_len = 512,
    pretrained_model_name_or_path = 'neuralmind/bert-base-portuguese-cased',    
    modelo_spacy = 'pt_core_news_lg',
    versao_spacy = '2.3.0',
    do_lower_case = False,   # default True
    output_attentions = False,    # default False
    output_hidden_states = True, # default False    
    estrategia_pooling = 0, # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
    palavra_relevante = 0 # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
)

# Constantes 
ESTRATEGIA_POOLING = ['MEAN','MAX']
PALAVRA_RELEVANTE = ['ALL', 'CLEAN', 'NOUN']

class CoherenceBERT:
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path):
        # Parâmetro recebido para o modelo BERT
        model_args.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # Carrega o modelo e tokenizador do BERT        
        self.model, self.tokenizer = carregaBERT(model_args)
        
        self.verificaCarregamentoSpacy()
    
    def verificaCarregamentoSpacy(self):
        ''' 
        Verifica se é necessário carregar o spacy.
        Utilizado para as estratégias de palavras relevantes CLEAN e NOUN.
        ''' 
        
        if model_args.palavra_relevante != 0:
            # Carrega o modelo spacy
            self.nlp = carregaSpacy(model_args)
            
        else:
            self.nlp = None
    
    def defineEstrategiaPooling(self, estrategiaPooling):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.
        ''' 
        
        if estrategiaPooling == ESTRATEGIA_POOLING[1]:
            model_args.estrategia_pooling = 1
            
        else:
            model_args.estrategia_pooling = 0

    def definePalavraRelevante(self, palavraRelevante):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        ''' 
        
        if palavraRelevante == PALAVRA_RELEVANTE[1]:
            model_args.palavra_relevante = 1
            verificaCarregamentoSpacy()
            
        else:
            if palavraRelevante == PALAVRA_RELEVANTE[2]:
                model_args.palavra_relevante = 2
                verificaCarregamentoSpacy()
                
            else:
                model_args.palavra_relevante = 0


    def getMedidaCoerencia(self, texto, estrategiaPooling = 'MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        ''' 

        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                    modelo=self.model, 
                                                    tokenizador=self.tokenizer, 
                                                    nlp=self.nlp, 
                                                    camada=listaTipoCamadas[4], 
                                                    tipoDocumento='o', 
                                                    estrategia_pooling=model_args.estrategia_pooling, 
                                                    palavra_relevante=model_args.palavra_relevante)
          
        return self.Ccos, self.Ceuc, self.Cman

    
    def getMedidaCoerenciaCosseno(self, texto, estrategiaPooling = 'MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de coerência do texto utilizando a medida de similaridade de cosseno.
        ''' 
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                    modelo=self.model, 
                                                    tokenizador=self.tokenizer, 
                                                    nlp=self.nlp, 
                                                    camada=listaTipoCamadas[4], 
                                                    tipoDocumento='o', 
                                                    estrategia_pooling=model_args.estrategia_pooling, 
                                                    palavra_relevante=model_args.palavra_relevante)
          
        return self.Ccos
    
    def getMedidaCoerenciaEuclediana(self, texto, estrategiaPooling = 'MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Euclidiana.
        ''' 
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                    modelo=self.model, 
                                                    tokenizador=self.tokenizer, 
                                                    nlp=self.nlp, 
                                                    camada=listaTipoCamadas[4], 
                                                    tipoDocumento='o', 
                                                    estrategia_pooling=model_args.estrategia_pooling, 
                                                    palavra_relevante=model_args.palavra_relevante)
          
        return self.Ceuc        
    
    def getMedidaCoerenciaManhattan(self, texto, estrategiaPooling = 'MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Manhattan.
        ''' 
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)
        
        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                    modelo=self.model, 
                                                    tokenizador=self.tokenizer, 
                                                    nlp=self.nlp, 
                                                    camada=listaTipoCamadas[4], 
                                                    tipoDocumento='o', 
                                                    estrategia_pooling=model_args.estrategia_pooling, 
                                                    palavra_relevante=model_args.palavra_relevante)
          
        return self.Cman                
