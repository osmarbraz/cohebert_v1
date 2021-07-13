# Import das bibliotecas.
import numpy as np
import torch

def getDocumentoLista(listaDocumento):
    '''
    Recebe uma lista de senten�as e faz a concatena��o em uma string
    '''

    stringDocumento = ''  
    # Concatena as senten�as do documento
    for sentenca in listaDocumento:                
        stringDocumento = stringDocumento + sentenca

def getListaSentencasDocumento(documento, nlp):

    '''
    Retorna uma lista com as senten�as de um documento. Utiliza o spacy para dividir o documento em senten�as.
    '''

    # Aplica tokeniza��o de senten�a do spacy no documento
    doc = nlp(documento) 

    # Lista para as senten�as
    lista = []
    # Percorre as senten�as
    for sentenca in doc.sents: 
        # Adiciona as senten�as a lista
        lista.append(str(sentenca))

    return lista

def encontrarIndiceSubLista(lista, sublista):
    '''
    Localiza os �ndices de in�cio e fim de uma sublista em uma lista
    '''
    # https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm
    h = len(lista)
    n = len(sublista)
    skip = {sublista[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if lista[i - j] != sublista[-j - 1]:
                i += skip.get(lista[i], n)
                break
        else:
            indiceInicio = i - n + 1
            indiceFim = indiceInicio + len(sublista)-1
            return indiceInicio, indiceFim
    return -1, -1

def removeStopWord(documento, stopwords):
    '''
    Remove as stopwords de um documento.
    '''

    # Remo��o das stop words do documento
    documentoSemStopwords = [palavra for palavra in documento.split() if palavra.lower() not in stopwords]

    # Concatena o documento sem os stopwords
    documentoLimpo = ' '.join(documentoSemStopwords)

    # Retorna o documento
    return documentoLimpo

def retornaSaliente(documento, tipoSaliente='NOUN', nlp):
    '''
    Retorna somente os palavras do documento ou senten�a do tipo especificado.
    '''
  
    # Realiza o parsing no spacy
    doc = nlp(documento)

    # Retorna a lista das palavras salientes
    documentoComSubstantivos = [token.text for token in doc if token.pos_ == tipoSaliente]

    # Concatena o documento com os substantivos
    documentoConcatenado = ' '.join(documentoComSubstantivos)

    # Retorna o documento
    return documentoConcatenado

def getDocumentoTokenizado(documento, tokenizador):

    '''
    Retorna um documento tokenizado e concatenado com tokens especiais '[CLS]' no in�cio
    e o token '[SEP]' no fim para ser submetido ao BERT.
    '''

    # Adiciona os tokens especiais.
    documentoMarcado = '[CLS] ' + documento + ' [SEP]'

    # Documento tokenizado
    documentoTokenizado = tokenizador.tokenize(documentoMarcado)

    return documentoTokenizado



# Constantes para padronizar o acesso aos dados do modelo do BERT.
TEXTO_TOKENIZADO = 0
INPUT_IDS = 1
ATTENTION_MASK = 2
TOKEN_TYPE_IDS = 3
OUTPUTS = 4
OUTPUTS_LAST_HIDDEN_STATE = 0
OUTPUTS_POOLER_OUTPUT = 1
OUTPUTS_HIDDEN_STATES = 2
 
def getEmbeddingsTodasCamadas(documento, modelo, tokenizador):
    
    '''   
    Retorna os embeddings de todas as camadas de um documento.
    '''

    # Documento tokenizado
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)

    #print('O documento (', documento, ') tem tamanho = ', len(documentoTokenizado), ' = ', documentoTokenizado)

    # Recupera a quantidade tokens do documento tokenizado.
    qtdeTokens = len(documentoTokenizado)

    #tokeniza o documento e retorna os tensores.
    dicCodificado = tokenizador.encode_plus(
                        documento,                          # Documento a ser codificado.
                        add_special_tokens = True,      # Adiciona os tokens especiais '[CLS]' e '[SEP]'
                        max_length = qtdeTokens,        # Define o tamanho m�ximo para preencheer ou truncar.
                        truncation = True,              # Trunca o documento por max_length
                        padding = 'max_length',         # Preenche o documento at� max_length
                        return_attention_mask = True,   # Constr�i a m�scara de aten��o.
                        return_tensors = 'pt'           # Retorna os dados como tensores pytorch.
                   )
    
    # Ids dos tokens de entrada mapeados em seus �ndices do vocabu�rio.
    input_ids =  dicCodificado['input_ids']

    # M�scara de aten��o de cada um dos tokens como pertencentes � senten�a '1'.
    attention_mask = dicCodificado['attention_mask']

    # Recupera os tensores dos segmentos.
    token_type_ids = dicCodificado['token_type_ids']

    # Roda o documento atrav�s do BERT, e coleta todos os estados ocultos produzidos.
    # das 12 camadas. 
    with torch.no_grad():

        # Passe para a frente, calcule as previs�es outputs.     
        outputs = modelo(input_ids=input_ids, 
                        attention_mask=attention_mask)

        # A avalia��o do modelo retorna um n�mero de diferentes objetos com base em
        # como � configurado na chamada do m�todo `from_pretrained` anterior. Nesse caso,
        # porque definimos `output_hidden_states = True`, o terceiro item ser� o
        # estados ocultos(hidden_states) de todas as camadas. Veja a documenta��o para mais detalhes:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

        # Retorno de model quando �output_hidden_states=True� � setado:    
        # outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
        # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    return documentoTokenizado, input_ids, attention_mask, token_type_ids, outputs

# getEmbeddingsTodasCamadasBuffer
# Cria um buffer com os embeddings de senten�as para economizar no processamento.
buffer_embeddings = {}

def getEmbeddingsTodasCamadasBuffer(S, modelo, tokenizador):
    '''
    Retorna os embeddings de uma senten�a de um buffer ou do modelo..
    '''
    # Se est� no dicion�rio retorna o embedding
    if S in buffer_embeddings:
        return buffer_embeddings.get(S)
    else:
        # Gera o embedding
        totalCamada = getEmbeddingsTodasCamadas(S, modelo, tokenizador)
        buffer_embeddings.update({S: totalCamada})
        return totalCamada

def limpaBufferEmbedding():
    '''
    Esvazia o buffer de embeddings das senten�as.
    '''
    buffer_embeddings.clear()


# getEmbeddingCamada
# Retorna os embeddings das camadas especificas.

def getEmbeddingPrimeiraCamada(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbeddinging � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a primeira(-1) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][0]
    # Sa�da: (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    #print('resultado=',resultado.size())

    return resultado

def getEmbeddingPenultimaCamada(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a pen�ltima(-2) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-2]
    # Sa�da: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('resultado=',resultado.size())

    return resultado

def getEmbeddingUltimaCamada(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a �ltima(-1) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-1]
    # Sa�da: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('resultado=',resultado.size())
  
    return resultado    

def getEmbeddingSoma4UltimasCamadas(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas as 4 �ltimas camadas
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    embeddingCamadas = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-4:]
    # Sa�da: List das camadas(4) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  

    # Usa o m�todo `stack` para criar uma nova dimens�o no tensor 
    # com a concate��o dos tensores dos embeddings.        
    # Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    resultadoStack = torch.stack(embeddingCamadas, dim=0)
    # Sa�da: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultadoStack=',resultadoStack.size())
  
    # Realiza a soma dos embeddings de todos os tokens para as camadas
    # Entrada: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    resultado = torch.sum(resultadoStack, dim=0)
    # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultado=',resultado.size())

    return resultado

def getEmbeddingConcat4UltimasCamadas(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
    
    # Cria uma lista com os tensores a serem concatenados
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    # Lista com os tensores a serem concatenados
    listaConcat = []
    # Percorre os 4 �ltimos
    for i in [-1,-2,-3,-4]:
        # Concatena da lista
        listaConcat.append(sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][i])
    # Sa�da: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('listaConcat=',len(listaConcat))

    # Realiza a concatena��o dos embeddings de todos as camadas
    # Sa�da: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    resultado = torch.cat(listaConcat, dim=-1)
    # Sa�da: Entrada: (<1(lote)> x <qtde_tokens> <3072 ou 4096>)  
    # print('resultado=',resultado.size())
  
    return resultado   

def getEmbeddingSomaTodasAsCamada(sentencaEmbedding):
    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas as camadas descontando a primeira(0)
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    embeddingCamadas = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][1:]
    # Sa�da: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  

    # Usa o m�todo `stack` para criar uma nova dimens�o no tensor 
    # com a concate��o dos tensores dos embeddings.        
    # Entrada: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultadoStack = torch.stack(embeddingCamadas, dim=0)
    # Sa�da: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultadoStack=',resultadoStack.size())
  
    # Realiza a soma dos embeddings de todos os tokens para as camadas
    # Entrada: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    resultado = torch.sum(resultadoStack, dim=0)
    # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
    # print('resultado=',resultado.size())
  
    return resultado


def getResultadoEmbeddings(sentencaEmbedding, camada):
    '''
    Retorna o resultado da opera��o sobre os embeddings das camadas de acordo com tipo de camada especificada.
    '''

    # Cada elemento do vetor sentencaEmbedding � formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states � uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>) 

    resultadoEmbeddingCamadas = None
  
    if camada[LISTATIPOCAMADA_ID] == PRIMEIRA_CAMADA:
        resultadoEmbeddingCamadas = getEmbeddingPrimeiraCamada(sentencaEmbedding)
        #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
    else:
        if camada[LISTATIPOCAMADA_ID] == PENULTIMA_CAMADA:
            resultadoEmbeddingCamadas = getEmbeddingPenultimaCamada(sentencaEmbedding)
            #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
        else:
            if camada[LISTATIPOCAMADA_ID] == ULTIMA_CAMADA:
                resultadoEmbeddingCamadas = getEmbeddingUltimaCamada(sentencaEmbedding)
                #print('resultadoEmbeddingCamadas2=',resultadoEmbeddingCamadas.size())
            else:
                if camada[LISTATIPOCAMADA_ID] == SOMA_4_ULTIMAS_CAMADAS:
                    resultadoEmbeddingCamadas = getEmbeddingSoma4UltimasCamadas(sentencaEmbedding)            
                    #print('resultadoEmbeddingCamadas3=',resultadoEmbeddingCamadas.size())
                else:
                    if camada[LISTATIPOCAMADA_ID] == CONCAT_4_ULTIMAS_CAMADAS:
                        resultadoEmbeddingCamadas = getEmbeddingConcat4UltimasCamadas(sentencaEmbedding)
                        #print('resultadoEmbeddingCamadas4=',resultadoEmbeddingCamadas.size())
                    else:
                        if camada[LISTATIPOCAMADA_ID] == TODAS_AS_CAMADAS:
                            resultadoEmbeddingCamadas = getEmbeddingSomaTodasAsCamada(sentencaEmbedding)
                            #print('resultadoEmbeddingCamadas5=',resultadoEmbeddingCamadas.size())
                            # Sa�da: <1> x <qtde_tokens> x <768 ou 1024>
  
    # Verifica se a primeira dimens�o � 1 para remover
    # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
    if resultadoEmbeddingCamadas.shape[0] == 1:
        # Remove a dimens�o 0 caso seja de tamanho 1.
        # Usa o m�todo 'squeeze' para remover a primeira dimens�o(0) pois possui tamanho 1
        # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
        resultadoEmbeddingCamadas = torch.squeeze(resultadoEmbeddingCamadas, dim=0)     
    #print('resultadoEmbeddingCamadas2=', resultadoEmbeddingCamadas.size())    
    # Sa�da: <qtde_tokens> x <768 ou 1024>
  
    # Retorna o resultados dos embeddings dos tokens da senten�a  
    return resultadoEmbeddingCamadas

def getMedidasSentencasEmbeddingMEAN(embeddingSi, embeddingSj):
    '''
    Retorna as medidas de duas senten�as Si e Sj utilizando a estrat�gia MEAN.
    - Entrada
        - embeddingSi - os embeddings da primeira senten�a
        - embeddingSj - os embeddings da segunda senten�a
    - Sa�da
        - Scos - Similaridade do coseno - usando a m�dia dos embeddings Si e Sj das camadas especificadas
        - Seuc - Dist�ncia euclidiana - usando a m�dia dos embeddings Si e Sj das camadas especificadas
        - Sman - Dist�ncia de manhattan - usando a m�dia dos embeddings Si e Sj das camadas especificadas
    '''

    #print('embeddingSi=', embeddingSi.shape) 
    #print('embeddingSj=', embeddingSj.shape)
  
    # As opera��es de subtra��o(sub), mul(multiplica��o/produto), soma(sum), cosseno(similaridade), euclediana(diferen�a) e manhattan(diferen�a)
    # Necessitam que os embeddings tenha a mesmo n�mero de dimens�es.
  
    # Calcula a m�dia dos embeddings para os tokens de Si, removendo a primeira dimens�o.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    mediaEmbeddingSi = torch.mean(embeddingSi, dim=0)    
    # Sa�da: <768 ou 1024>
    #print('mediaCamadasSi=', mediaCamadasSi.shape)
  
    # Calcula a m�dia dos embeddings para os tokens de Sj, removendo a primeira dimens�o.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    mediaEmbeddingSj = torch.mean(embeddingSj, dim=0)    
    # Sa�da: <768 ou 1024>
    #print('mediaCamadasSj=', mediaCamadasSj.shape)
    
    # Similaridade do cosseno entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Scos = similaridadeCoseno(mediaEmbeddingSi, mediaEmbeddingSj)
    # Sa�da: N�mero real

    # Dist�ncia euclidiana entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Seuc = distanciaEuclidiana(mediaEmbeddingSi, mediaEmbeddingSj)
    # Sa�da: N�mero real

    # Dist�ncia de manhattan entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Sman = distanciaManhattan(mediaEmbeddingSi, mediaEmbeddingSj)
    # Sa�da: N�mero real

    # Retorno das medidas das senten�as  
    return mediaEmbeddingSi, mediaEmbeddingSj, Scos, Seuc, Sman


def getMedidasSentencasEmbeddingMAX(embeddingSi, embeddingSj):
     '''
     Retorna as medidas de duas senten�as Si e Sj utilizando a estrat�gia MAX.
     - Entrada
        - embeddingSi - os embeddings da primeira senten�a
        - embeddingSj - os embeddings da segunda senten�a

     - Sa�da
        - Scos - Similaridade do coseno - usando o maior dos embeddings Si e Sj das camadas especificadas
        - Seuc - Dist�ncia euclidiana - usando o maior dos embeddings Si e Sj das camadas especificadas
        - Sman - Dist�ncia de manhattan - usando o maior dos embeddings Si e Sj das camadas especificadas
    '''

    #print('embeddingSi=', embeddingSi.shape) 
    #print('embeddingSj=', embeddingSj.shape)

    # As opera��es de subtra��o(sub), mul(multiplica��o/produto), soma(sum), cosseno(similaridade), euclediana(diferen�a) e manhattan(diferen�a)
    # Necessitam que os embeddings tenha a mesmo n�mero de dimens�es.

    # Encontra os maiores embeddings os tokens de Si, removendo a primeira dimens�o.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    maiorEmbeddingSi, linha = torch.max(embeddingSi, dim=0)    
    # Sa�da: <768 ou 1024>
    #print('maiorEmbeddingSi=', maiorEmbeddingSi.shape)

    # Encontra os maiores embeddings os tokens de Sj, removendo a primeira dimens�o.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    maiorEmbeddingSj, linha = torch.max(embeddingSj, dim=0)    
    # Sa�da: <768 ou 1024>
    #print('maiorEmbeddingSj=', maiorEmbeddingSj.shape)

    # Similaridade do cosseno entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Scos = similaridadeCoseno(maiorEmbeddingSi, maiorEmbeddingSj)
    # Sa�da: N�mero real

    # Dist�ncia euclidiana entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Seuc = distanciaEuclidiana(maiorEmbeddingSi, maiorEmbeddingSj)
    # Sa�da: N�mero real

    # Dist�ncia de manhattan entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Sman = distanciaManhattan(maiorEmbeddingSi, maiorEmbeddingSj)
    # Sa�da: N�mero real

    # Retorno das medidas das senten�as
    return maiorEmbeddingSi, maiorEmbeddingSj, Scos, Seuc, Sman


def getMedidasSentencasEmbedding(embeddingSi, embeddingSj, estrategiaPooling):
    '''
    Realiza o c�lculo da medida do documento de acordo com a estrat�gia.
    '''

    if estrategiaPooling == 0:
        return getMedidasSentencasEmbeddingMEAN(embeddingSi, embeddingSj)
    else:
        return getMedidasSentencasEmbeddingMAX(embeddingSi, embeddingSj)


def getEmbeddingSentencaEmbeddingDocumentoComTodasPalavras(embeddingDocumento, documento, sentenca, tokenizador):
    '''
    Retorna os embeddings de uma senten�a com todas as palavras a partir dos embeddings do documento.
    '''
        
    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)
    #print(documentoTokenizado)

    # Tokeniza a senten�a
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)
    #print(sentencaTokenizada)
    # Remove os tokens de in�cio e fim da senten�a
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')    
    #print(len(sentencaTokenizada))

    # Localiza os �ndices dos tokens da senten�a no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado,sentencaTokenizada)
    #print(inicio,fim) 

    # Recupera os embeddings dos tokens da senten�a a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim+1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Retorna o embedding da senten�a no documento
    return embeddingSentenca

def getEmbeddingSentencaEmbeddingDocumentoSemStopWord(embeddingDocumento, documento, sentenca, tokenizador):
    '''
    Retorna os embeddings de uma senten�a sem stopwords a partir dos embeddings do documento.
    '''
      
    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)  
    #print(documentoTokenizado)

    # Remove as stopword da senten�a
    sentencaSemStopWord = removeStopWord(sentenca, spacy_stopwords)

    # Tokeniza a senten�a sem stopword
    sentencaTokenizadaSemStopWord = getDocumentoTokenizado(sentencaSemStopWord, tokenizador)
    #print(sentencaTokenizadaSemStopWord)

    # Remove os tokens de in�cio e fim da senten�a
    sentencaTokenizadaSemStopWord.remove('[CLS]')
    sentencaTokenizadaSemStopWord.remove('[SEP]')    
    #print(len(sentencaTokenizadaSemStopWord))

    # Tokeniza a senten�a
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)

    # Remove os tokens de in�cio e fim da senten�a
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')  
    #print(sentencaTokenizada)
    #print(len(sentencaTokenizada))

    # Localiza os �ndices dos tokens da senten�a no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado,sentencaTokenizada)
    #print('Senten�a inicia em:', inicio, 'at�', fim) 

    # Recupera os embeddings dos tokens da senten�a a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim+1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Lista com os tensores selecionados
    listaTokensSelecionados = []
    # Localizar os embeddings dos tokens da senten�a tokenizada sem stop word na senten�a 
    # Procura somente no intervalo da senten�a
    for i, tokenSenten�a in enumerate(sentencaTokenizada):
        for tokenSentencaSemStopWord in sentencaTokenizadaSemStopWord: 
            if tokenSenten�a == tokenSentencaSemStopWord:
                #embeddingSentencaSemStopWord = torch.cat((embeddingSentencaSemStopWord, embeddingSentenca[i:i+1]), dim=0)
                listaTokensSelecionados.append(embeddingSentenca[i:i+1])

    embeddingSentencaSemStopWord = None

    if len(listaTokensSelecionados) != 0:
        # Concatena os vetores da lista pela dimens�o 0
        embeddingSentencaSemStopWord = torch.cat(listaTokensSelecionados, dim=0)
        #print("embeddingSentencaSemStopWord:",embeddingSentencaSemStopWord.shape)

    # Retorna o embedding da senten�a no documento
    return embeddingSentencaSemStopWord

def getEmbeddingSentencaEmbeddingDocumentoSomenteSaliente(embeddingDocumento, documento, sentenca, tokenizador, tipoSaliente='NOUN'):
    '''
    Retorna os embeddings de uma senten�a somente com as palavras salientes a partir dos embeddings do documento.
    '''

    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)  
    #print(documentoTokenizado)

    # Retorna as palavras salientes da senten�a do tipo especificado
    sentencaSomenteSaliente = retornaSaliente(sentenca,tipoSaliente)

    # Tokeniza a senten�a 
    sentencaTokenizadaSomenteSaliente = getDocumentoTokenizado(sentencaSomenteSaliente, tokenizador)

    # Remove os tokens de in�cio e fim da senten�a
    sentencaTokenizadaSomenteSaliente.remove('[CLS]')
    sentencaTokenizadaSomenteSaliente.remove('[SEP]')  
    #print(sentencaTokenizadaSomenteSaliente)
    #print(len(sentencaTokenizadaSomenteSaliente))

    # Tokeniza a senten�a
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)

    # Remove os tokens de in�cio e fim da senten�a
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')  
    #print(sentencaTokenizada)
    #print(len(sentencaTokenizada))

    # Localiza os �ndices dos tokens da senten�a no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado,sentencaTokenizada)
    #print('Senten�a inicia em:', inicio, 'at�', fim) 

    # Recupera os embeddings dos tokens da senten�a a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim+1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Lista com os tensores selecionados
    listaTokensSelecionados = []
    # Localizar os embeddings dos tokens da senten�a tokenizada sem stop word na senten�a 
    # Procura somente no intervalo da senten�a
    for i, tokenSentenca in enumerate(sentencaTokenizada):
        for tokenSentencaSomenteSaliente in sentencaTokenizadaSomenteSaliente: 
            if tokenSentenca == tokenSentencaSomenteSaliente:        
                listaTokensSelecionados.append(embeddingSentenca[i:i+1])

    embeddingSentencaComSubstantivo = None

    if len(listaTokensSelecionados) != 0:
        # Concatena os vetores da lista pela dimens�o 0  
        embeddingSentencaComSubstantivo = torch.cat(listaTokensSelecionados, dim=0)
        #print("embeddingSentencaComSubstantivo:",embeddingSentencaComSubstantivo.shape)

    # Retorna o embedding da senten�a do documento
    return embeddingSentencaComSubstantivo

def getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, documento, sentenca, tokenizador, filtro=0):
    '''
    Retorna os embeddings de uma senten�a com todas as palavras, sem stopwords ou somente com palavra substantivas a partir dos embeddings do documentos.
    '''

    if filtro == 0:
        return getEmbeddingSentencaEmbeddingDocumentoComTodasPalavras(embeddingDocumento, documento, sentenca, tokenizador)
    else:
        if filtro == 1:
            return getEmbeddingSentencaEmbeddingDocumentoSemStopWord(embeddingDocumento, documento, sentenca, tokenizador)
        else:
            if filtro == 2:
                return getEmbeddingSentencaEmbeddingDocumentoSomenteSaliente(embeddingDocumento, documento, sentenca, tokenizador, tipoSaliente='NOUN')



def getMedidasCoerenciaDocumento(documento, modelo, tokenizador, camada, tipoDocumento='p', filtro=0):
    '''
    Retorna as medidas de coer�ncia do documento.
    Considera somente senten�as com alguma palavra.
    '''

    # Quantidade de senten�as no documento
    n = len(documento)
    # Divisor da quantidade de documentos
    divisor = n - 1

    # Documento � uma lista com as senten�as
    #print('camada=',camada)
    #print('Documento=', documento)

    # Junta a lista de senten�as em um documento(string)
    stringDocumento = ' '.join(documento)

    # Envia o documento ao MCL e recupera os embeddings de todas as camadas
    # Se for o documento original pega do buffer para evitar a repeti��o
    if tipoDocumento == 'o':
        # Retorna os embeddings de todas as camadas do documento
        # O embedding possui os seguintes valores        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        totalCamadasDocumento =  getEmbeddingsTodasCamadasBuffer(stringDocumento, modelo, tokenizador)      
        # Sa�da: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 
    else:
        # Retorna os embeddings de todas as camadas do documento
        # O embedding possui os seguintes valores        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        totalCamadasDocumento =  getEmbeddingsTodasCamadas(stringDocumento, modelo, tokenizador)      
        # Sa�da: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 

    # Recupera os embeddings dos tokens das camadas especificadas de acordo com a estrat�gia especificada para camada  
    embeddingDocumento = getResultadoEmbeddings(totalCamadasDocumento, camada=camada)
    #print('embeddingDocumento=', embeddingDocumento.shape)

    # Acumuladores das medidas entre as senten�as  
    somaScos = 0
    somaSeuc = 0
    somaSman = 0

    # Seleciona os pares de senten�a a serem avaliados
    posSi = 0
    posSj = posSi + 1

    #Enquanto o ind�ce da sentne�a posSj(2a senten�a) n�o chegou ao final da quantidade de senten�as
    while posSj <= (n-1):  

        # Seleciona as senten�as do documento  
        Si = documento[posSi]
        Sj = documento[posSj]

        # Recupera os embedding das senten�as Si e Sj do embedding do documento      
        embeddingSi = getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, stringDocumento, Si, tokenizador, filtro=filtro)                                
        embeddingSj = getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, stringDocumento, Sj, tokenizador, filtro=filtro)

        # Verifica se os embeddings senten�as est�o preenchidos
        if embeddingSi != None and embeddingSj != None:

          # Recupera as medidas entre Si e Sj     
          ajustadoEmbeddingSi, ajustadoEmbeddingSj, Scos, Seuc, Sman = getMedidasSentencasEmbedding(embeddingSi, embeddingSj)

          # Acumula as medidas
          somaScos = somaScos + Scos
          somaSeuc = somaSeuc + Seuc
          somaSman = somaSman + Sman

           # avan�a para o pr�ximo par de senten�as
          posSi = posSj
          posSj = posSj + 1
        else:
          # Reduz um da quantidade de senten�as pois uma delas est� vazia
          divisor = divisor - 1
          # Se embeddingSi igual a None avanca pos1 e pos2
          if embeddingSi == None:
            # Avan�a a posi��o da senten�a posSi para a posSj
            posSi = posSj
            # Avan�a para a pr�xima senten�a de posSj
            posSj = posSj + 1        
          else:          
            # Se embeddingSj = None avan�a somente posJ para a pr�xima senten�a
            if embeddingSj == None:
              posSj = posSj + 1

    # Calcula a medida 
    Ccos =  0
    Ceuc =  0
    Cman =  0

    if divisor != 0:
      Ccos = float(somaScos)/float(divisor)
      Ceuc = float(somaSeuc)/float(divisor)
      Cman = float(somaSman)/float(divisor)

    return Ccos, Ceuc, Cman


# listaTipoCamadas
# Define uma lista com as camadas a serem analisadas nos teste.
# Cada elemento da lista 'listaTipoCamadas' � chamado de camada sendo formado por:
#  - camada[0] = �ndice da camada
#  - camada[1] = Um inteiro com o �ndice da camada a ser avaliada. Pode conter valores negativos.
#  - camada[2] = Opera��o para n camadas, CONCAT ou SUM.
#  - camada[3] = Nome do tipo camada

# Os nomes do tipo da camada pr�-definidos.
#  - 0 - Primeira                    
#  - 1 - Pen�ltima
#  - 2 - �ltima
#  - 3 - Soma 4 �ltimas
#  - 4 - Concat 4 �ltimas
#  - 5 - Todas

# Constantes para facilitar o acesso os tipos de camadas
PRIMEIRA_CAMADA = 0
PENULTIMA_CAMADA = 1
ULTIMA_CAMADA = 2
SOMA_4_ULTIMAS_CAMADAS = 3
CONCAT_4_ULTIMAS_CAMADAS = 4
TODAS_AS_CAMADAS = 5

# �ndice dos campos da camada
LISTATIPOCAMADA_ID = 0
LISTATIPOCAMADA_CAMADA = 1
LISTATIPOCAMADA_OPERACAO = 2
LISTATIPOCAMADA_NOME = 3

# BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# O �ndice da camada com valor positivo indica uma camada espec�fica
# O �ndica com um valor negativo indica as camadas da posi��o com base no fim descontado o valor indice at� o fim.
listaTipoCamadas = [
                      [ PRIMEIRA_CAMADA,           1, '-',      'Primeira'],                      
                      [ PENULTIMA_CAMADA,         -2, '-',      'Pen�ltima'],
                      [ ULTIMA_CAMADA,            -1, '-',      '�ltima'],
                      [ SOMA_4_ULTIMAS_CAMADAS,   -4, 'SUM',    'Soma 4 �ltimas'],
                      [ CONCAT_4_ULTIMAS_CAMADAS, -4, 'CONCAT', 'Concat 4 �ltimas'],                      
                      [ TODAS_AS_CAMADAS,          24, 'SUM',    'Todas']
                    ]

# listaTipoCamadas e suas refer�ncias:
# 0 - Primeira            listaTipoCamadas[PRIMEIRA_CAMADA]
# 1 - Pen�ltima           listaTipoCamadas[PENULTIMA_CAMADA]
# 2 - �ltima              listaTipoCamadas[ULTIMA_CAMADA]
# 3 - Soma 4 �ltimas      listaTipoCamadas[SOMA_4_ULTIMAS_CAMADAS]
# 4 - Concat 4 �ltimas    listaTipoCamadas[CONCAT_4_ULTIMAS_CAMADAS]
# 5 - Todas               listaTipoCamadas[TODAS_AS_CAMADAS]


def comparaMedidasCamadasSentencas(Si, Sj, modelo, tokenizador, camada):
    '''
    Facilita a exibi��o dos valores de compara��o de duas ora��es.
    '''
  
    # Recupera os embeddings da senten�a 1 e senten�a 2
    embeddingSi, embeddingSj, Scos, Seuc, Sman = getMedidasCamadasSentencas(Si, Sj, modelo, tokenizador, camada)

    print('  ->Mostra compara��o da ' + camada[LISTATIPOCAMADA_NOME]+ ' camada(s)')    
    print('   Cosseno(SixSj)     = %.8f' % Scos)
    print('   Euclidiana(SixSj)  = %.8f' % Seuc)
    print('   Manhattan(SixSj)   = %.8f' % Sman)