# Import das bibliotecas.
import logging  # Biblioteca de logging
import os  # Biblioteca para apagar arquivos
import datetime # Biblioteca de data e hora
import pandas as pd # Biblioteca para manipulação e análise de dados

# Import de bibliotecas próprias
from bert.bertmodulo  import *

# ===================================================================================
# Módulo para agrupar as operações específicas de classificação binária para 
# a validação cruzada Kfold.
# ===================================================================================

# ============================
def carregaResultadoAvaliacao(model_args, training_args, DIRETORIO_AVALIACAO):
    '''
    Carrega e mostra os dados da avaliação de folds. 
    
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo. 
        `training_args` - Objeto com os argumentos do treinamento. 
        `DIRETORIO_AVALIACAO` - Diretório com os dados da avaliação.        .        
    '''      
    
    # Acumuladores.
    somaAcuracia = 0
    listaTempo = []
    contaFolds = 0

    # Verifica o nome do modelo BERT a ser utilizado
    MODELO_BERT = getNomeModeloBERT(model_args)

    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT =  getTamanhoBERT(model_args)
    
    # Nome arquivo resultado avaliação
    NOME_ARQUIVO_AVALIACAO = training_args.output_dir

    logging.info("Média dos arquivos: {} X {}.".format(NOME_ARQUIVO_AVALIACAO, TAMANHO_BERT))

    # Verifica se o diretório dos resultados existem.
    if os.path.exists(DIRETORIO_AVALIACAO):
        # Percorre os arquivos de resultados.
        for f in range(10):  
            # Nome do arquivo a ser aberto.
            NOME_ARQUIVO_AVALIACAO_COMPLETO = DIRETORIO_AVALIACAO + NOME_ARQUIVO_AVALIACAO + str(f+1) + MODELO_BERT + TAMANHO_BERT + ".csv"    
            # Verifica se o arquivo existe.
            if os.path.isfile(NOME_ARQUIVO_AVALIACAO_COMPLETO):
                # Carrega os dados do arquivo  
                dados = pd.read_csv(NOME_ARQUIVO_AVALIACAO_COMPLETO, sep=';')
                # Mostra os dados do teste do fold.
                for index, linha in dados.iterrows():
        
                    # Cálculo das estatísticas
                    acc = (linha['vp']+linha['vn'])/(linha['vp']+linha['vn']+linha['fp']+linha['fn'])
                
                    if (linha['vp']+linha['fn']) != 0:
                        rec = (linha['vp'])/(linha['vp']+linha['fn'])
                    else:
                        rec = 0
                    if (linha['vp']+linha['fp']) != 0:
                        pre = (linha['vp'])/(linha['vp']+linha['fp'])
                    else:  
                        pre = 0
                    if (pre + rec) != 0:  
                        f1 = 2 * ((pre * rec)/(pre + rec))
                    else:
                        f1 = 0
                    
                    qtdeTestes = linha['vp']+linha['vn']+linha['fp']+linha['fn']
                    
                    logging.info("Arquivo: {}, Data: {}, Tempo: {}, QtdeTeste: {:3d}, Acc: {:.8f}, Rec: {:.8f}, Pre: {:.8f}, F1:{:.8f}, vp: {:4d}; vn: {:4d}; fp: {:4d}; fn: {:4d}".format(
                        linha['arquivo'], linha['data'], linha['tempo'], qtdeTestes, acc, rec, pre, f1, linha['vp'], linha['vn'], linha['fp'], linha['fn']))    
           
                    # Guarda o tempo.
                    listaTempo.append(str(linha['tempo']))

                # Procura a maior acurácia.
                somaAcuracia = somaAcuracia + dados['acuracia'].max()
                # Conta o número de folds.
                contaFolds = contaFolds + 1
    
        # Mostra a soma da acurácia . 
        logging.info("Total acurácia                                       : {:.8f}.".format(somaAcuracia))
        # Mostra a quantidade de folds.
        logging.info("Quantidade de folds                                  : {}.".format(contaFolds))  
        # Calcula a média.
        media = somaAcuracia/contaFolds
        logging.info("A média da acurácia de {:2d} folds é                    : {:.8f}.".format(contaFolds, media))
        logging.info("O tempo gasto na execução do treinamentoa {:2d} folds é : {}.".format(contaFolds, somaTempo(listaTempo)))
        logging.info("A média de tempo de execução de {:2d} folds é           : {}.".format(contaFolds, mediaTempo(listaTempo)))
    else:
        logging.info('Diretório com os resultados não encontrado.')
        
# ============================        
def salvaResultadoClassificacao(model_args, training_args, DIRETORIO_CLASSIFICACAO, lista_resultado_avaliacao):
    '''
    Salva o resultado da classificação de um fold. 
    
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo.
        `training_args` - Objeto com os argumentos do treinamento. 
        `DIRETORIO_CLASSIFICACAO` - Diretório para salvar os dados da classificação.
        `lista_resultado_avaliacao` - Lista com os dados da avaliação da classificação. 
    '''  

    if model_args.salvar_classificacao:
        
        # Recupera a hora do sistema.
        data_e_hora = datetime.datetime.now()
        
        # Verifica o nome do modelo BERT a ser utilizado
        MODELO_BERT = getNomeModeloBERT(model_args)

        # Verifica o tamanho do modelo(default large)
        TAMANHO_BERT =  getTamanhoBERT(model_args)

        # Nome arquivo resultado
        NOME_ARQUIVO_CLASSIFICACAO = training_args.output_dir + str(model_args.fold) + MODELO_BERT + TAMANHO_BERT 
      
        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO_CLASSIFICACAO):  
            # Cria o diretório
            os.makedirs(DIRETORIO_CLASSIFICACAO)
            logging.info("Diretório criado: {}.".format(DIRETORIO_CLASSIFICACAO))
            
        else:
            logging.info("Diretório já existe: {}.".format(DIRETORIO_CLASSIFICACAO))

        # Nome do arquivo a ser aberto.
        NOME_ARQUIVO_CLASSIFICACAO_COMPLETO = DIRETORIO_CLASSIFICACAO + NOME_ARQUIVO_CLASSIFICACAO + ".csv"

        # Cabeçalho do arquivo csv
        CABECALHO_ARQUIVO = "data;id;classe;predicao"
        
        # Gera todo o conteúdo a ser salvo no arquivo        
        novoConteudo = ""        
        for resultado in lista_resultado_avaliacao:      
            novoConteudo = novoConteudo + data_e_hora.strftime("%d/%m/%Y %H:%M") + ";" + str(resultado[0]) + ";" + str(resultado[1]) + ";" + str(resultado[2]) + "\n"

        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO):
            logging.info("Atualizando arquivo classificação: {}.".format(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO))
            # Abre o arquivo para leitura.
            arquivo = open(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO,'r')
            # Leitura de todas as linhas do arquivo.
            conteudo = arquivo.readlines()
            # Conteúdo a ser adicionado.
            conteudo.append(novoConteudo)

            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO,'w')
            # escreva o conteúdo criado anteriormente nele.
            arquivo.writelines(CABECALHO_ARQUIVO + '\n' + conteudo)  
            # Fecha o arquivo.
            arquivo.close()
            
        else:
            logging.info("Criando arquivo classificação: {}.".format(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO))
            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_CLASSIFICACAO_COMPLETO,'w')
            arquivo.writelines(CABECALHO_ARQUIVO + '\n' + novoConteudo)  # escreva o conteúdo criado anteriormente nele.
            # Fecha o arquivo.
            arquivo.close()        

# ============================        
def salvaResultadoAvaliacao(model_args, training_args, DIRETORIO_AVALIACAO, acc, rec, pre, f1, vp_s, vn_s, fp_s, fn_s):
    '''
    Salva os dados da avaliação de um fold. 
    
    Parâmetros:
        `model_args` - Objeto com os argumentos do modelo. 
        `training_args` - Objeto com os argumentos do treinamento. 
        `DIRETORIO_AVALIACAO` - Diretório para salvar os dados da avaliação.        .        
    '''      

    if model_args.salvar_avaliacao:

        # Recupera a hora do sistema.
        data_e_hora = datetime.datetime.now()
        
        # Verifica o nome do modelo BERT a ser utilizado
        MODELO_BERT = getNomeModeloBERT(model_args)

        # Verifica o tamanho do modelo(default large)
        TAMANHO_BERT =  getTamanhoBERT(model_args)

        # Nome arquivo resultado
        NOME_ARQUIVO_AVALIACAO = training_args.output_dir + str(model_args.fold) + MODELO_BERT + TAMANHO_BERT 

        # Verifica se o diretório existe
        if not os.path.exists(DIRETORIO_AVALIACAO):  
            # Cria o diretório
            os.makedirs(DIRETORIO_AVALIACAO)
            logging.info("Diretório criado: {}.".format(DIRETORIO_AVALIACAO))            
        else:
            logging.info("Diretório já existe: {}.".format(DIRETORIO_AVALIACAO))

        # Cabeçalho do arquivo csv
        CABECALHO_ARQUIVO = "arquivo;data;tempo;acuracia;vp;vn;fp;fn"
            
        # Nome do arquivo a ser aberto.
        NOME_ARQUIVO_AVALIACAO_COMPLETO = DIRETORIO_AVALIACAO + NOME_ARQUIVO_AVALIACAO + ".csv"

        # Conteúdo a ser adicionado.
        novoConteudo = NOME_ARQUIVO_AVALIACAO + ";" +  data_e_hora.strftime("%d/%m/%Y %H:%M") + ";"  + treinamento_total + ";"  + str(acc) + ";"  +  str(vp_s) + ";"  +  str(vn_s) + ";" +  str(fp_s) + ";" +  str(fn_s) + "\n"
        
        # Verifica se o arquivo existe.
        if os.path.isfile(NOME_ARQUIVO_AVALIACAO_COMPLETO):
            logging.info("Atualizando arquivo resultado: {}.".format(NOME_ARQUIVO_AVALIACAO_COMPLETO))
            # Abre o arquivo para leitura.
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO,'r')
            # Leitura de todas as linhas do arquivo.
            conteudo = arquivo.readlines()
            # Conteúdo a ser adicionado.
            conteudo.append(novoConteudo)

            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO,'w')
            # escreva o conteúdo criado anteriormente nele.
            arquivo.writelines(conteudo)  
            # Fecha o arquivo.
            arquivo.close()
            
        else:
            logging.info("Criando arquivo resultado: {}.".format(NOME_ARQUIVO_AVALIACAO_COMPLETO))
            # Abre novamente o arquivo (escrita).
            arquivo = open(NOME_ARQUIVO_AVALIACAO_COMPLETO,'w')
            arquivo.writelines(CABECALHO_ARQUIVO + '\n' + novoConteudo)  # escreva o conteúdo criado anteriormente nele.
            # Fecha o arquivo.
            arquivo.close()
