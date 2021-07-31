# Import das bibliotecas.
import logging  # Biblioteca de logging
from tqdm.notebook import tqdm as tqdm_notebook # Biblioteca da barra de progresso
import os  # Biblioteca para apagar arquivos
import datetime # Biblioteca de data e hora
import pandas as pd # Biblioteca para manipulação e análise de dados

# Import de bibliotecas próprias
from medidor.medidor import *

# ===================================================================================
# Módulo para agrupar as operações de calculo das medidas de (in)coerência.
# ===================================================================================

# ============================
def getSomatorioDiferencaAbsolutaOrdenada(listaMedida1, listaMedida2):
    '''
    Calcula o somatório da diferença absoluta entre a lista ordenada de duas medidas.
    
    Parâmetros:
        `listaMedida1` - Lista 1 com medidas.
        `listaMedida2` - Lista 2 com medidas.
        
    Saída:  
        `soma` - Somátorio da diferença absoluta.
    '''  
    lista1 = sorted(listaMedida1)
    lista2 = sorted(listaMedida2)
    soma = 0
    for i, linha in enumerate(lista1): 
        diferenca = abs(lista2[i] - lista1[i])
        soma = soma + diferenca
        
    return soma

# ============================
def getSomatorioDiferencaAbsoluta(listaMedida1, listaMedida2):
    '''
    Calcula o somatório da diferença absoluta entre a lista de duas medidas.
    
    Parâmetros:
        `listaMedida1` - Lista 1 com medidas.
        `listaMedida2` - Lista 2 com medidas.
        
    Saída:  
        `soma` - Somátorio da diferença absoluta.
    '''  
    
    soma = 0
    for i, linha in enumerate(listaMedida1): 
        diferenca = abs(listaMedida2[i] - listaMedida1[i])
        soma = soma + diferenca
        
    return soma

# ============================
def acertosMedidaSimilaridadePermutado(medida, dfListaParesDocumentosMedidas):
    '''
    Conta os acerto de uma medida de similaridade(DO < permDO) com base em documentos permutados. 
    
    Parâmetros:
        `medida` - Medida a ser contada.
        `dfListaParesDocumentosMedidas` - Datafrane dos documentos e suas medidas.
        
    Saída:  
        `acertosOriginal` - Quantidade de acertos do documento original para a medida.
        `acertosPermutado` - Quantidade de acertos do documento permutado para a medida.
        `percentualOriginal` - Percentual de acertos do documento original para a medida.
        `percentualPermutado` - Percentual de acertos do documento original para a medida.        
    '''  
    
    acertosOriginal  = 0
    acertosPermutado  = 0  
    for i, linha in dfListaParesDocumentosMedidas.iterrows():
        if linha[medida + 'DO']  < linha[medida + 'Perm']:
            acertosOriginal = acertosOriginal + 1
            
        else:
            acertosPermutado = acertosPermutado + 1

    percentualOriginal = acertosOriginal / len(dfListaParesDocumentosMedidas)
    percentualPermutado = acertosPermutado / len(dfListaParesDocumentosMedidas)

    return acertosOriginal, acertosPermutado, percentualOriginal, percentualPermutado  

# ============================
def acertosMedidaSimilaridadeOriginal(medida, dfListaParesDocumentosMedidas):
    '''
    Conta os acerto de uma medida de similaridade(DO >= permDO) com base em documentos originais. 
    
    Parâmetros:
        `medida` - Medida a ser contada.
        `dfListaParesDocumentosMedidas` - Datafrane dos documentos e suas medidas.
        
    Saída:  
        `acertosOriginal` - Quantidade de acertos do documento original para a medida.
        `acertosPermutado` - Quantidade de acertos do documento permutado para a medida.
        `percentualOriginal` - Percentual de acertos do documento original para a medida.
        `percentualPermutado` - Percentual de acertos do documento original para a medida.        
    '''  
        
    acertosOriginal  = 0
    acertosPermutado  = 0  
    for i, linha in dfListaParesDocumentosMedidas.iterrows():
        if linha[medida + 'DO']  >= linha[medida + 'Perm']:
            acertosOriginal = acertosOriginal + 1
            
        else:
            acertosPermutado = acertosPermutado + 1

    percentualOriginal = acertosOriginal / len(dfListaParesDocumentosMedidas)
    percentualPermutado = acertosPermutado / len(dfListaParesDocumentosMedidas)

    return acertosOriginal, acertosPermutado, percentualOriginal, percentualPermutado  

# ============================
def acertosMedidaDistanciaPermutado(medida, dfListaParesDocumentosMedidas):
    '''
    Conta os acerto de uma medida de distância(DO < permDO) com base em documentos permutados. 
    
    Parâmetros:
        `medida` - Medida a ser contada.
        `dfListaParesDocumentosMedidas` - Datafrane dos documentos e suas medidas.
        
    Saída:  
        `acertosOriginal` - Quantidade de acertos do documento original para a medida.
        `acertosPermutado` - Quantidade de acertos do documento permutado para a medida.
        `percentualOriginal` - Percentual de acertos do documento original para a medida.
        `percentualPermutado` - Percentual de acertos do documento original para a medida.        
    '''  
    
    acertosOriginal  = 0  
    acertosPermutado  = 0  
    for i, linha in dfListaParesDocumentosMedidas.iterrows():
        if linha[medida + 'DO'] < linha[medida + 'Perm']:      
            acertosPermutado = acertosPermutado + 1
            
        else:
            acertosOriginal = acertosOriginal + 1
      
    percentualOriginal = acertosOriginal / len(dfListaParesDocumentosMedidas)
    percentualPermutado = acertosPermutado / len(dfListaParesDocumentosMedidas)

    return acertosOriginal, acertosPermutado, percentualOriginal, percentualPermutado  

# ============================
def acertosMedidaDistanciaOriginal(medida, dfListaParesDocumentosMedidas):
    '''
    Conta os acerto de uma medida de distância(DO <= permDO) com base em documentos originais. 
    
    Parâmetros:
        `medida` - Medida a ser contado.
        `dfListaParesDocumentosMedidas` - Datafrane dos documentos e suas medidas.
        
    Saída:  
        `acertosOriginal` - Quantidade de acertos do documento original para a medida.
        `acertosPermutado` - Quantidade de acertos do documento permutado para a medida.
        `percentualOriginal` - Percentual de acertos do documento original para a medida.
        `percentualPermutado` - Percentual de acertos do documento original para a medida.        
    '''  
    acertosOriginal  = 0  
    acertosPermutado  = 0  
    
    for i, linha in dfListaParesDocumentosMedidas.iterrows():
        if linha[medida + 'DO'] <= linha[medida + 'Perm']:
            acertosOriginal = acertosOriginal + 1
            
        else:
            acertosPermutado = acertosPermutado + 1

    percentualOriginal = acertosOriginal / len(dfListaParesDocumentosMedidas)
    percentualPermutado = acertosPermutado / len(dfListaParesDocumentosMedidas)

    return acertosOriginal, acertosPermutado, percentualOriginal, percentualPermutado

# ============================
def recuperaListasDeMedidas(medida, dfListaParesDocumentosMedidas):
    '''
    Divide o dataframe uma lista de documentos originais e uma lista de documentos permutados para uma medida.
    Parâmetros:
        `medida` - Medida a ser recuperada.
        `dfListaParesDocumentosMedidas` - Datafrane dos documentos e suas medidas.
        
    Saída:     
        `lista_medida_original` - Lista com os documentos originais da medida.
        `lista_medida_permutado` - Lista com os documentos permutados da medida.
    '''        

    # Medida do documento original 
    lista_medida_original = [linha[medida + 'DO'] for i, linha in dfListaParesDocumentosMedidas.iterrows()]
    
    # Medida do documento permutado
    lista_medida_permutado = [linha[medida + 'Perm'] for i, linha in dfListaParesDocumentosMedidas.iterrows()]

    return lista_medida_original, lista_medida_permutado

# ============================
def geraEstatisticasMedidasDocumentos(dfdadosMedidasDocumentos):
    '''
    Gera as estatísticas dos pares dos documentos e suas medidas.
    Parâmetros:
        `dfdadosMedidasDocumentos` - Datafrane dos documentos e suas medidas.
        
    Saída:     
        `stats_medidas_documentos` - Lista com as estatísticas.        
    '''        

    # Lista das estatísticas das medidas
    stats_medidas_documentos = []

    for i, linha in dfdadosMedidasDocumentos.iterrows():
      
        # Registra as estatística da comparação
        stats_medidas_documentos.append(
                                        {
                                        'documento': i, 
                                        'original ccos': linha['ccosDO'],
                                        'permutado ccos': linha['ccosPerm'],
                                        'dif ccos': linha['ccosDO'] - linha['ccosPerm'],
                                        'difabs ccos': abs(linha['ccosDO'] - linha['ccosPerm']),
                                        'original ceuc': linha['ceucDO'],
                                        'permutado ceuc': linha['ceucPerm'],
                                        'dif ceuc': linha['ceucDO'] - linha['ceucPerm'],
                                        'difabs ceuc': abs(linha['ceucDO'] - linha['ceucPerm']),
                                        'original cman': linha['cmanDO'],
                                        'permutado cman': linha['cmanPerm'],
                                        'dif cman': linha['cmanDO'] - linha['cmanPerm'],
                                        'difabs cman': abs(linha['cmanDO'] - linha['cmanPerm']),
                                        }
                                        )     
        
    return stats_medidas_documentos

# ============================
def separaDocumentos(dadosMedida):
    '''
    Separa os dados do dataframe em originais e permutados.
    Parâmetros:
        `dadosMedida` - Dados a serem separados em originais e permutados.    
        
    Saída:
        `dfOriginalMedida` - Dataframe com os dados de documentos originais.        
        `dfPermutadoMedida` - Dataframe com os dados de documentos permutados.
    '''        
    # Separa os originais
    dfOriginalMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm') == False]
    # Remove os duplicados
    dfOriginalMedida = dfOriginalMedida.drop_duplicates(subset=['arquivo'])
    logging.info("Registros: {}.".format(len(dfOriginalMedida)))

    # Separa os permutados
    dfPermutadoMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm') == True]
    # Remove os duplicados
    dfPermutadoMedida = dfPermutadoMedida.drop_duplicates(subset=['arquivo'])
    
    logging.info("Registros: {}.".format(len(dfPermutadoMedida)))

    return dfOriginalMedida, dfPermutadoMedida

# ============================
def salvaResultadoMedicao(model_args, NOME_BASE, DIRETORIO_MEDICAO, lista_medidas_documentos_salvar):

    if model_args.salvar_medicao:

        # Recupera a hora do sistema.
        data_e_hora = datetime.datetime.now()

        AJUSTADO = '_pretreinado'
        if model_args.usar_mcl_ajustado == True:
            AJUSTADO = '_ajustado'

        ESTRATEGIA_POOLING = '_mean'
        if model_args.estrategia_pooling == 1:
            ESTRATEGIA_POOLING = '_max'

        PALAVRA_RELEVANTE = '_tap'
        if model_args.palavra_relevante == 1:
            PALAVRA_RELEVANTE = '_ssw'       
        else:
            if model_args.palavra_relevante == 2:
                PALAVRA_RELEVANTE = '_ssb'   

        # Nome arquivo resultado
        NOME_ARQUIVO_MEDICAO = NOME_BASE + AJUSTADO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + getNomeModeloBERT(model_args) + getTamanhoBERT(model_args)

        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO_MEDICAO):  
            # Cria o diretório
            os.makedirs(DIRETORIO_MEDICAO)
            logging.info("Diretório criado: {}.".format(DIRETORIO_MEDICAO))
    
        else:
            logging.info("Diretório já existe: {}.".format(DIRETORIO_MEDICAO))

        # Nome do arquivo a ser aberto.
        NOME_ARQUIVO_MEDICAO_COMPLETO = DIRETORIO_MEDICAO + NOME_ARQUIVO_MEDICAO + '.csv'
    
        # Cabeçalho do arquivo csv
        CABECALHO_ARQUIVO = "data;arquivo;ccos;ceuc;cman"

        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO_MEDICAO):  
            # Cria o diretório
            os.makedirs(DIRETORIO_MEDICAO)
            logging.info("Diretório criado: {}.".format(DIRETORIO_MEDICAO))
    
        else:
            logging.info("Diretório já existe: {}.".format(DIRETORIO_MEDICAO))

        # Nome do arquivo a ser aberto.
        NOME_ARQUIVO_MEDICAO_COMPLETO = DIRETORIO_MEDICAO + NOME_ARQUIVO_MEDICAO + '.csv'

        # Gera todo o conteúdo a ser salvo no arquivo
        novoConteudo = ''        
        for resultado in lista_medidas_documentos_salvar:            
            novoConteudo = novoConteudo + data_e_hora.strftime('%d/%m/%Y %H:%M') + ';' + str(resultado[0]) + ';' + str(resultado[1]) + ';'  + str(resultado[2]) + ';'  + str(resultado[3]) + '\n'

        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_MEDICAO_COMPLETO):
            logging.info("Atualizando arquivo medição: {}.".format(NOME_ARQUIVO_MEDICAO_COMPLETO))
            # Abre o arquivo para leitura.
            arquivo = open(NOME_ARQUIVO_MEDICAO_COMPLETO, 'r')
            # Leitura de todas as linhas do arquivo.
            conteudo = arquivo.readlines()
            # Conteúdo a ser adicionado.
            conteudo.append(novoConteudo)

            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_MEDICAO_COMPLETO, 'w')
            # escreva o conteúdo criado anteriormente nele.
            arquivo.writelines(conteudo)  
            # Fecha o arquivo.
            arquivo.close()
        else:
        
            logging.info("Criando arquivo medição: {}.".format(NOME_ARQUIVO_MEDICAO_COMPLETO))
            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_MEDICAO_COMPLETO, 'w')
            arquivo.writelines(CABECALHO_ARQUIVO + '\n' + novoConteudo)  # escreva o conteúdo criado anteriormente nele.
            # Fecha o arquivo.
            arquivo.close()

# ============================        
def salvaResultadoAvaliacao(model_args, NOME_BASE, DIRETORIO_AVALIACAO, tempoTotalProcessamento, conta, acuraciaCcos, contaCcos, acuraciaCeuc, contaCeuc, acuraciaCman, contaCman):

    if model_args.salvar_avaliacao:

        # Recupera a hora do sistema.
        data_e_hora = datetime.datetime.now()

        AJUSTADO = '_pretreinado'
        if model_args.usar_mcl_ajustado == True:
            AJUSTADO = '_ajustado'

        ESTRATEGIA_POOLING = '_mean'
        if model_args.estrategia_pooling == 1:
            ESTRATEGIA_POOLING = '_max'

        PALAVRA_RELEVANTE = '_tap'
        if model_args.palavra_relevante == 1:
            PALAVRA_RELEVANTE = '_ssw'       
        else:
            if model_args.palavra_relevante == 2:
                PALAVRA_RELEVANTE = '_ssb' 
    
        # Nome arquivo resultado
        NOME_ARQUIVO_AVALIACAO = NOME_BASE + AJUSTADO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + getNomeModeloBERT(model_args) + getTamanhoBERT(model_args)

        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO_AVALIACAO):  
            # Cria o diretório
            os.makedirs(DIRETORIO_AVALIACAO)
            logging.info("Diretório criado: {}".format(DIRETORIO_AVALIACAO))
        else:
            logging.info("Diretório já existe: {}".format(DIRETORIO_AVALIACAO))

        # Nome do arquivo a ser aberto.
        NOME_ARQUIVO_AVALIACAO_COMPLETO = DIRETORIO_AVALIACAO + NOME_ARQUIVO_AVALIACAO + '.csv'
    
        # Cabeçalho do arquivo csv
        CABECALHO_ARQUIVO = "arquivo;data;tempo;conta;ccos;contaccos;ceuc;contaceuc;cman;contacman"

        # Conteúdo a ser adicionado.
        novoConteudo = NOME_ARQUIVO_AVALIACAO + ';' + data_e_hora.strftime('%d/%m/%Y %H:%M') + ';' + tempoTotalProcessamento + ';'  + str(conta) + ';'  + str(acuraciaCcos) + ';' + str(contaCcos) + ';' + str(acuraciaCeuc) + ';' + str(contaCeuc) + ';' + str(acuraciaCman) + ';' + str(contaCman) + '\n'

        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_AVALIACAO_COMPLETO):
            logging.info("Atualizando arquivo resultado avaliação: {}.".format(NOME_ARQUIVO_AVALIACAO_COMPLETO))
            # Abre o arquivo para leitura.
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO, 'r')
            # Leitura de todas as linhas do arquivo.
            conteudo = arquivo.readlines()
            # Conteúdo a ser adicionado.
            conteudo.append(novoConteudo)

            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO, 'w')
            # escreva o conteúdo criado anteriormente nele.
            arquivo.writelines(conteudo)  
            # Fecha o arquivo.
            arquivo.close()
    
        else:
            logging.info("Criando arquivo resultado avaliação: {}.".format(NOME_ARQUIVO_AVALIACAO_COMPLETO))
            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO, 'w')
            arquivo.writelines(CABECALHO_ARQUIVO + '\n' + novoConteudo)  # escreva o conteúdo criado anteriormente nele.
            # Fecha o arquivo.
            arquivo.close()

# ============================
def carregaMedidas(NOME_BASE, DIRETORIO_MEDIDAS, TIPO_MODELO, ESTRATEGIA_POOLING, PALAVRA_RELEVANTE, NOME_MODELO_BERT, TAMANHO_BERT):
    '''
    Carrega as medidas de coerência de um diretório e retorna um dataframe.
    Parâmetros:
        `NOME_BASE` - Nome base do arquivo de medidas.    
        `DIRETORIO_MEDIDAS` - Diretório com os arquivos das medidas.    
        `TIPO_MODELO` - Tipo do modelo(pretreinado ou ajustado) a ser carregado.  
        `ESTRATEGIA_POOLING` - Nome da estratégia de pooling(MEAN ou MAX).
        `PALAVRA_RELEVANTE` - Nome da estratégia de relevância(ALL, CLEAN ou NOUN).
        `NOME_MODELO_BERT` - Nome do modelo(BERTimbau ou BERT) a ser carregado.  
        `TAMANHO_BERT` - Tamanho do modelo(Base ou Large) a ser carregado. 
        
    Saída:
        `dfMedida` - Um dataframe com os dados carregados.
    '''
        
    NOME_ARQUIVO_MEDICAO = NOME_BASE + TIPO_MODELO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + NOME_MODELO_BERT + TAMANHO_BERT + '.csv'

    dfMedida = None
     
    # Verifica se o diretório dos resultados existem.
    if os.path.exists(DIRETORIO_MEDIDAS):
        arquivos = os.listdir(DIRETORIO_MEDIDAS)     

        NOME_ARQUIVO_MEDICAO_COMPLETO = DIRETORIO_MEDIDAS + NOME_ARQUIVO_MEDICAO
    
        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_MEDICAO_COMPLETO):
            logging.info("Carregando arquivo: {}.".format(NOME_ARQUIVO_MEDICAO))
            
            # Carrega os dados do arquivo  
            dfMedida = pd.read_csv(NOME_ARQUIVO_MEDICAO_COMPLETO, sep=';')
            
            logging.info("Medidas carregadas: {}.".format(len(dfMedida)))
      
        else:
            logging.info("Arquivo com as medições não encontrado!")        

    else:
        logging.info("Diretório com as medições não encontrado!")

    return dfMedida

# ============================
def calculaMedidasDocumentosConjuntoDeDados(model_args, dfdados, model, tokenizer, nlp, wandb):
    '''
    Cálcula a medida de todos os documentos do conjunto.
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo. 
        `dfdados` - Datafrane dos documentos.
        `model` - Modelo BERT.
        `tokenizer` - Tokenizador BERT.
        `nlp` - Objeto spaCy.        
        `wandg` - Wandb para log do experimento.  
        
    Saída:     
        `lista_medidas_documentos_salvar` - Lista com as medidas dos documentos para salvamento.
        `conta` - Quantidade de pares de documentos.
        `percentualCcos` - Percentual de acertos para a medida Ccos.
        `contaCcos` - Quantidade de acertos para a medida Ccos.
        `percentualCeuc` - Percentual de acertos para a medida Ceuc
        `contaCeuc` - Quantidade de acertos para a medida Ceuc
        `percentualCman` - Percentual de acertos para a medida Cman
        `contaCman` - Quantidade de acertos para a medida Cman
                
    '''  
    
    # Contadores de ocorrência de coerência
    contaCcos = 0
    contaCeuc = 0
    contaCman = 0
    conta = 0

    # Lista para o salvamento das medidas
    lista_medidas_documentos_salvar = []

    # Barras de progresso.    
    dfdado_bar = tqdm_notebook(dfdados.iterrows(), desc=f"Pares documentos", unit=f"par", total=len(dfdados))

    # Percorre as pares de documento carregadas do arquivo
    for (i, linha) in dfdado_bar:
   
        # Conta os pares
        conta = conta + 1

        # Calcula as medidas do documento original    
        original = linha['sentencasOriginais']    
        Ccos, Ceuc, Cman = getMedidasCoerenciaDocumento(original, modelo=model, tokenizador=tokenizer, nlp=nlp, camada=listaTipoCamadas[4], tipoDocumento='o', estrategia_pooling=model_args.estrategia_pooling, palavra_relevante=model_args.palavra_relevante)
                      
        # Calcula a smedidas do documento permutado
        permutado = linha['sentencasPermutadas']
        Ccosp, Ceucp, Cmanp = getMedidasCoerenciaDocumento(permutado, modelo=model, tokenizador=tokenizer, nlp=nlp, camada=listaTipoCamadas[4], tipoDocumento='p', estrategia_pooling=model_args.estrategia_pooling, palavra_relevante=model_args.palavra_relevante)
      
        # Verifica a medida de coerência Scos(similaridade do cosseno) das sentenças do documento original com as sentenças do documento permutado.
        # Quanto maior o valor de Scos mais as orações do documentos são coerentes
        if Ccos >= Ccosp:
            contaCcos = contaCcos + 1

        # Verifica a medida de incoerência Seuc(distância euclidiana) das sentenças do documento original com as sentenças do documento permutado.
        # Quanto maior o valor de Scos mais as orações do documentos são coerentes
        if Ceuc <= Ceucp:
            contaCeuc = contaCeuc + 1

        # Verifica a medida de incoerência Sman(distância de manhattan) das sentenças do documento original com as sentenças do documento permutado.
        # Quanto maior o valor de Scos mais as orações do documentos são coerentes
        if Cman <= Cmanp:
            contaCman = contaCman + 1        

        # Guarda as medidas em uma lista para salvar em arquivo
        # Guarda as medidas dos documentos originais
        lista_medidas_documentos_salvar.append([linha[0], Ccos, Ceuc, Cman])
        # Guarda as medidas dos documentos permutados
        lista_medidas_documentos_salvar.append([linha[3], Ccosp, Ceucp, Cmanp])

    logging.info("Total de Pares: {}.".format(conta))
    
    if model_args.use_wandb:
        wandb.log({'pares_doc': conta})

    logging.info("Pares Corretos Ccos: {}.".format(contaCcos))
    percentualCcos = float(contaCcos) / float(conta)
    logging.info("Percentual acertos Ccos: {}.".format(percentualCcos * 100))

    if model_args.use_wandb:
        wandb.log({'acuracia_ccos': acuraciaCcos})

    logging.info("Pares Corretos Ceuc: {}.".format(contaCeuc))
    percentualCeuc = float(contaCeuc) / float(conta)
    logging.info("Percentual acertos Ceuc: {}.".format(percentualCeuc * 100))

    if model_args.use_wandb:
        wandb.log({'acuracia_ceuc': acuraciaCeuc})  

    logging.info("Pares Corretos Cman: {}.".format(contaCman))
    percentualCman = float(contaCman) / float(conta)
    logging.info("Percentual acertos Cman: {}.".format(percentualCman * 100))

    if model_args.use_wandb:
        wandb.log({'acuracia_cman': acuraciaCman})  

    logging.info("Cálculo de medida de documentos terminado!")

    return lista_medidas_documentos_salvar, conta, percentualCcos, contaCcos, percentualCeuc, contaCeuc, percentualCman, contaCman

# ============================
def organizaParesDocumentos(dfOriginalMedida, dfPermutadoMedida):
    '''
    Organiza as medidas do.
    Parâmetros:
        `dfOriginalMedida` - Dados de medidas de documentos originais.
        `dfPermutadoMedida` - Dados de medidas de documentos permutados.
        
    Saída:
        `dfListaParesDocumentosMedidas` - Dataframe com as medidas de documentos originais seguidos das medidas de documentos permutados.
    '''  

    listaParesDocumentosMedidas = []
    # Refaz os pares de documentos
    for i, linha1 in dfOriginalMedida.iterrows():
        ponto = linha1['arquivo'].find('.')
        nomeArquivo = linha1['arquivo'][:ponto]
        
        for i, linha2 in dfPermutadoMedida.iterrows():
            if nomeArquivo in linha2['arquivo']:
                listaParesDocumentosMedidas.append(
                                                   [linha1['data'], 
                                                   linha1['arquivo'],	
                                                   linha1['ccos'], 
                                                   linha1['ceuc'], 
                                                   linha1['cman'],
                                       
                                                   linha2['data'], 
                                                   linha2['arquivo'],	
                                                   linha2['ccos'], 
                                                   linha2['ceuc'], 
                                                   linha2['cman']], )

    logging.info("Registros antes: {}.".format(len(listaParesDocumentosMedidas)))

    dfListaParesDocumentosMedidas = pd.DataFrame(listaParesDocumentosMedidas, columns=('dataDO', 'idDO', 'ccosDO', 'ceucDO', 'cmanDO', 'dataPerm', 'idPerm', 'ccosPerm', 'ceucPerm', 'cmanPerm'))   
    logging.info("Registros depois: {}.".format(len(dfListaParesDocumentosMedidas)))

    return dfListaParesDocumentosMedidas
