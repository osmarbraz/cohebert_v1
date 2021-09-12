# Import das bibliotecas.
import logging  # Biblioteca de logging
import pandas as pd # Biblioteca pandas

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from conjuntodedados.dadosonlineeducmedida import *

# ============================
def divisaoConjuntoDados(dfdados, percentualDivisao=0.3, classeStratify='classe'):
    '''    
    Divide o conjunto de dados em treino e teste utilizando um percentual de divisão.
    
    Parâmetros:
    `dfdados` - Dataframe com os dados a serem divididos.  
    `percentualDivisao` - Percentual de divisão dos dados.
    `classeStratify` - Faz uma divisão de forma que a proporção dos valores na amostra produzida seja a mesma que a proporção dos valores fornecidos.
    
    Retorno:
    `dfdados_train` - Dataframe com os dados de treinamento.
    `dfdados_test` - Dataframe com os dados de teste.
    '''
        
    # Quantidade de elementos de teste considerando o percentual
    test_qtde = int(percentualDivisao * dfdados.shape[0])
    
    # Divide o conjunto
    dfdados_train, dfdados_test = train_test_split(dfdados, test_size=test_qtde, random_state=42, stratify=dfdados[classeStratify])

    logging.info("Conjunto total: {}.".format(len(dfdados)))
    logging.info("  Treino: {}.".format(len(dfdados_train)))
    logging.info("  Teste : {}.".format(len(dfdados_test)))

    return dfdados_train, dfdados_test

# ============================
def organizaDados(dfdados):
    '''    
    Organiza osdados do OnlineEduc 1.0 para classificação e retorna um dataframe.
    Coloca os dados dos pares de documento um após o outro. 
    Primeiro adiciona o original e rotula como 1 e depois coloca o permutado rotulando como 0.
    
    Parâmetros:
    `dfdados` - Dataframe com os dados a serem organizados para classificação.  
    '''
  
    # Organiza os dados
    dados_organizados = []

    # Coloca o par um embaixo do outro.
    for index, linha in dfdados.iterrows():        
        # 1 Para original
        dados_organizados.append((linha['idOriginal'], linha['documentoOriginal'], 1))    
        # 0 para uma permutação 
        dados_organizados.append((linha['idPermutado'], linha['documentoPermutado'], 0))

    # Cria um dataframe com os dados
    dfdados = pd.DataFrame(dados_organizados, columns=["id", "documento", "classe"])      

    return dfdados 

# ============================   
def descartandoDocumentosGrandesClassificacao(model_args, tokenizer, dfdados):
    '''    
    Remove os documentos que extrapolam 512 tokens.
    Você pode definir o tamanho de documento que quiser no BERT, mas o modelo pré-treinado vem com um tamanho pré-definido. 
    No nosso caso vamos utilizar o modelo BERT, que tem 512 tokens de tamanho limite de documento. 
    O tokenizador gera quantidades diferentes tokens para cada modelo pré-treinado. 
    Portanto é necessário especificar o tokenizador para descatar os documentos que ultrapassam o limite de tokens de entrada do BERT.
    
    Parâmetros:              
    `model_args` - Objeto com os argumentos do modelo.    
    `tokenizer` - Tokenizador BERT.
    `dfdados` - Dataframe com os documentos a serem analisados.   
    
    Retorno:
    `dfdadosretorno` - Um dataframe sem os documentos grandes.
    '''
    
    dfdadosretorno = None
    
    # Verifica se o tokenizador foi carregado
    if tokenizer != None:
        
        # Remove colunas desnecessárias
        dfdados = dfdados.drop(columns=['sentencasOriginais', 'sentencasPermutadas'])
    
        # Define o tamanho máximo para os tokens
        tamanho_maximo = model_args.max_seq_len
  
        # Tokenize a codifica os documentos para o BERT.     
        dfdados['input_ids'] = dfdados['documentoOriginal'].apply(lambda tokens: tokenizer.encode(tokens, add_special_tokens=True))

        # Reduz para o tamanho máximo suportado pelo BERT.
        dfdados_512 = dfdados[dfdados['input_ids'].apply(len) <= tamanho_maximo]

        # Remove as colunas desnecessárias.
        dfdadosAnterior = dfdados.drop(columns=['input_ids'])
        dfdadosretorno = dfdados_512.drop(columns=['input_ids'])

        logging.info("Quantidade de dados anterior: {}.".format(len(dfdadosAnterior)))
        logging.info("Nova quantidade de dados    : {}.".format(len(dfdadosretorno)))

        # Registros removidos
        df = dfdadosAnterior.merge(dfdadosretorno, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        logging.info("Quantidade de registros removidos: {}.".format(len(df)))
        
    else:
        logging.info("Tokenizador não definido.")        

    return dfdadosretorno  

# ============================  
def getConjuntoDeDadosClassificacao(model_args, tokenizer): 
    '''    
    Carrega os dados do OnlineEduc 1.0 para classificação e retorna um dataframe.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.  
    `tokenizer` - Tokenizador BERT.
    
    Retorno:
    `dfdados` - Um dataframe com os dados carregados.
    '''
    
    # Realiza o download do conjunto de dados
    downloadConjuntoDeDados()
        
    # Carrega os pares de documentos dos arquivos
    lista_documentos = carregaParesDocumentosOnlineEduc()
        
    # Converte em um dataframe
    dfdados = converteListaParesDocumentos(lista_documentos)
        
    # Descarta os documentos muito grandes. (Que geram mais de 512 tokens)
    dfdados = descartandoDocumentosGrandesClassificacao(model_args, tokenizer, dfdados)
    
    # Organiza os dados para classificação
    dfdados = organizaDados(dfdados)
    
    return dfdados  
