# Import das bibliotecas.
import logging  # Biblioteca de logging
from tqdm.notebook import tqdm as tqdm_notebook # Biblioteca da barra de progresso
import os  # Biblioteca para apagar arquivos
import pandas as pd # Biblioteca para manipulação e análise de dados

# Import de bibliotecas próprias
from medidor.medidor import *


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
              'dif ccos' : linha['ccosDO'] - linha['ccosPerm'],
              'difabs ccos' : abs(linha['ccosDO'] - linha['ccosPerm']),
              'original ceuc': linha['ceucDO'],
              'permutado ceuc': linha['ceucPerm'],
              'dif ceuc' : linha['ceucDO'] - linha['ceucPerm'],
              'difabs ceuc' : abs(linha['ceucDO'] - linha['ceucPerm']),
              'original cman': linha['cmanDO'],
              'permutado cman': linha['cmanPerm'],
              'dif cman' : linha['cmanDO'] - linha['cmanPerm'],
              'difabs cman' : abs(linha['cmanDO'] - linha['cmanPerm']),
            }
        )     
        
    return stats_medidas_documentos

# ============================
def calculaMedidasDocumentosConjuntoDeDados(dfdados, model, tokenizer, nlp, model_args, wandb):
    '''
    Cálcula a medida de todos os documentos do conjunto.
    Parâmetros:
        `dfdados` - Datafrane dos documentos.
        `model` - Modelo BERT.
        `tokenizer` - Tokenizador BERT.
        `nlp` - Objeto spaCy.
        `model_args` - Objeto com os argumentos do modelo. 
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
    dfdado_bar = tqdm_notebook(dfdados.iterrows(), desc=f'Pares documentos', unit=f'par', total=len(dfdados))

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
        lista_medidas_documentos_salvar.append([linha[0], Ccos,  Ceuc,  Cman])
        # Guarda as medidas dos documentos permutados
        lista_medidas_documentos_salvar.append([linha[3], Ccosp, Ceucp, Cmanp])

    logging.info("Total de Pares: {}.".format(conta))
    
    if model_args.use_wandb:
        wandb.log({'pares_doc': conta})

    logging.info("Pares Corretos Ccos: {}.".format(contaCcos))
    percentualCcos = float(contaCcos)/float(conta)
    logging.info("Percentual acertos Ccos: {}.".format(percentualCcos*100))

    if model_args.use_wandb:
        wandb.log({'acuracia_ccos': acuraciaCcos})

    logging.info("Pares Corretos Ceuc: {}.".format(contaCeuc))
    percentualCeuc = float(contaCeuc)/float(conta)
    logging.info("Percentual acertos Ceuc: {}.".format(percentualCeuc*100))

    if model_args.use_wandb:
        wandb.log({'acuracia_ceuc': acuraciaCeuc})  

    logging.info("Pares Corretos Cman:",str(contaCman))
    percentualCman = float(contaCman)/float(conta)
    logging.info("Percentual acertos Cman: {}.".format(percentualCman*100))

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
                                                    linha2['cman'] ],)

    logging.info("Registros antes: {}.".format(len(listaParesDocumentosMedidas)))

    dfListaParesDocumentosMedidas = pd.DataFrame(listaParesDocumentosMedidas, columns=('dataDO', 'idDO', 'ccosDO', 'ceucDO', 'cmanDO', 'dataPerm', 'idPerm', 'ccosPerm', 'ceucPerm', 'cmanPerm'))   
    logging.info("Registros depois: {}.".format(len(dfListaParesDocumentosMedidas)))

    return dfListaParesDocumentosMedidas

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
    dfOriginalMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm')==False]
    # Remove os duplicados
    dfOriginalMedida = dfOriginalMedida.drop_duplicates(subset=['arquivo'])
    logging.info("Registros: {}.".format(len(dfOriginalMedida)))

    # Separa os permutados
    dfPermutadoMedida = dadosMedida.loc[dadosMedida['arquivo'].str.contains('Perm')==True]
    # Remove os duplicados
    dfPermutadoMedida = dfPermutadoMedida.drop_duplicates(subset=['arquivo'])
    
    logging.info("Registros: {}.".format(len(dfPermutadoMedida)))

    return dfOriginalMedida, dfPermutadoMedida

# ============================
def carregaMedidas(DIRETORIO_MEDIDAS, TIPO_MODELO, ESTRATEGIA_POOLING, PALAVRA_RELEVANTE, NOME_MODELO_BERT, TAMANHO_BERT):
    '''
    Carrega as medidas de coerência de um diretório e retorna um dataframe.
    Parâmetros:
        `DIRETORIO_MEDIDAS` - Diretório com os arquivos das medidas.    
        `TIPO_MODELO` - Tipo do modelo(pretreinado ou ajustado) a ser carregado.  
        `ESTRATEGIA_POOLING` - Nome da estratégia de pooling(MEAN ou MAX).
        `PALAVRA_RELEVANTE` - Nome da estratégia de relevância(ALL, CLEAN ou NOUN).
        `NOME_MODELO_BERT` - Nome do modelo(BERTimbau ou BERT) a ser carregado.  
        `TAMANHO_BERT` - Tamanho do modelo(Base ou Large) a ser carregado. 
        
    Saída:
        `dfMedida` - Um dataframe com os dados carregados.
    '''
    NOME_BASE = "MedicaoCoerenciaCSTNews_v1"
    
    NOME_ARQUIVO_MEDICAO = NOME_BASE + TIPO_MODELO + ESTRATEGIA_POOLING + PALAVRA_RELEVANTE + NOME_MODELO_BERT + TAMANHO_BERT + '.csv'

    dfMedida = None
     
    # Verifica se o diretório dos resultados existem.
    if os.path.exists(DIRETORIO_MEDIDAS):
        arquivos = os.listdir(DIRETORIO_MEDIDAS)     

        NOME_ARQUIVO_MEDICAO_COMPLETO = DIRETORIO_MEDIDAS + NOME_ARQUIVO_MEDICAO
    
        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_MEDICAO_COMPLETO):
            logging.info("Carregando arquivo:", NOME_ARQUIVO_MEDICAO)
            
            # Carrega os dados do arquivo  
            dfMedida = pd.read_csv(NOME_ARQUIVO_MEDICAO_COMPLETO, sep=';')
            
            logging.info('Medidas carregadas: ', len(dfMedida))
      
        else:
            logging.info('Arquivo com as medições não encontrado!')        

    else:
        logging.info('Diretório com as medições não encontrado!')

    return dfMedida
