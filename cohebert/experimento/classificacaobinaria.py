# Import das bibliotecas.
import logging  # Biblioteca de logging
import time  # Biblioteca de tempo
import random # Biblioteca para números aleatórios
import numpy as np # Biblioteca para manipulação e análise de dados
import torch # Biblioteca para manipular os tensores
from tqdm.notebook import tqdm as tqdm_notebook # Biblioteca para barra de progresso

from transformers import AdamW # Biblioteca do otimizador
from transformers import get_linear_schedule_with_warmup # Biblioteca do agendador

# Import de bibliotecas próprias
from bert.bertmodulo  import *

# ============================
def carregaOtimizador(training_args):
    '''
    Esta função carrega o otimizador utilizado no agendador de aprendizado.
    Parâmetros:
        `training_args` - Objeto com os argumentos do treinamento. 
    '''
    # Nota: AdamW é uma classe da biblioteca huggingface (ao contrário de pytorch).
    # Eu acredito que o 'W' significa 'Correção de redução de peso "
    otimizador = AdamW(model.parameters(),
                  lr = training_args.learning_rate, # (ou alfa) A taxa de aprendizado a ser usada. - default é 3e-5
                  # betas = (0.9, 0.999), # (beta1, beta2) - default é (0.9, 0.999)
                    # beta1 é taxa de decaimento exponencial para as estimativas do primeiro momento. 
                    # beta2 é taxa de decaimento exponencial para as estimativas do segundo momento. Este valor deve ser definido próximo a 1,0 em problemas com gradiente esparso (por exemplo, PNL e problemas de visão de computacional)
                  # eps = 1e-6, #  É um número muito pequeno para evitar qualquer divisão por zero na implementação - default é 1e-6.
                  # weight_decay = 0.0, # Correção de redução de peso. - default é 0.0
                    # A redução da taxa de aprendizagem também pode ser usada com Adam. A taxa de decaimento é atualizada a cada época para a demonstração da regressão logística.
                  # correct_bias = True #  Se não deve corrigir o viés(bias) no Adam mudar para False.- default é True
                )
  
    return otimizador
  
  
# ============================
def carregaAgendador(training_args, otimizador):

    '''
    Esta função carrega o agendador com um taxa de aprendizado que diminua linearmente até 0.
    Parâmetros:
        `training_args` - Objeto com os argumentos do treinamento. 
        `otimizador` - Objeto do otmizador do modelo. 
    '''

    # O número total de etapas de ajuste fino é [número de lotes] x [número de épocas].
    # (Observe que este não é o mesmo que o número de amostras de ajuste fino).
    total_etapas = len(documentos_treino) * training_args.num_train_epochs

    #Cria o agendador de taxa de aprendizagem.
    agendador = get_linear_schedule_with_warmup(otimizador, # O otimizador para o qual agendar a taxa de aprendizado.
                                            num_warmup_steps = 0, # O número de etapas para a fase de aquecimento. Valor default value em run_glue.py
                                            num_training_steps = total_etapas) # O número total de etapas de treinamento.


    logging.info("Total de etapas: {}".format(total_etapas))

    return agendador  

# ============================
def realizaAvaliacao(model_args, model, documentos_teste, classes_teste, documentoids_teste, wandb):

    # Armazena o resultado da avaliação executada
    lista_resultado_avaliacao = []

    print("\nRealizando Avaliação fold: {}".format(model_args.fold))

    # Predição no conjunto de teste no modelo.
    print('Predizendo rótulos para {:,} documentos de teste...'.format(len(documentos_teste)))

    # Use nossa nova função para preparar completamente nosso conjunto de dados.  
    (py_input_ids, py_attention_masks, py_labels, documentoids) = cria_lotes_inteligentes(model_args, tokenizer, documentos_teste, classes_teste, documentoids_teste, training_args.per_device_eval_batch_size)

    # Escolha um intervalo para imprimir atualizações de progresso.
    intervalo_atualizacao = obter_intervalo_atualizacao(total_iteracoes=len(py_input_ids), numero_atualizacoes=10)

    # Coloque o modelo em modo de avaliação.
    model.eval()

    # Acumula as perdas.
    test_losses = []

    totalacuracia = 0

    # Acumula os resultados dos testes.
    vp = [] # Verdadeiro positivo
    vn = [] # Verdadeiro negativo
    fp = [] # Falso positivo
    fn = [] # Falso negativo

    # Barra de progresso dos lotes de teste.
    lote_teste_bar = tqdm_notebook(range(0, len(py_input_ids)), desc=f'Lotes ', unit=f'lotes', total=len(py_input_ids))

    # Para cada lote dos dados de avaliação(teste).
    for index in lote_teste_bar:

        # Progresso é atualizado a cada lotes, por exemplo, 100 lotes.
        if index % intervalo_atualizacao == 0 and not index == 0:        
            # Calcula o tempo gasto em minutos.
            tempoGasto = formataTempo(time.time() - avaliacao_t0)
        
        # Calcula o tempo restante baseado no progresso.
        passos_por_segundo = (time.time() - avaliacao_t0) / index
        segundos_restantes = passos_por_segundo * (len(py_input_ids) - index)
        tempoRestante = formataTempo(segundos_restantes)

        # Mostra o progresso.
        print('  Lote {:>7,}  de  {:>7,}.    Gasto: {:}.  Restando: {:}'.format(index, len(py_input_ids), tempoGasto, tempoRestante))
    
        # Copia o lote para a GPU.
        d_input_ids = py_input_ids[index].to(device)
        d_input_mask = py_attention_masks[index].to(device)
        d_labels = py_labels[index].to(device)
        d_documentoids = documentoids[index]

        # Diga a pytorch para não se preocupar em construir o gráfico de computação durante
        # o passe para frente, já que isso só é necessário para backprop (treinamento).
        with torch.no_grad():
            # Obtenha a saída de "logits" pelo modelo. Os "logits" são a saída
            # valores antes de aplicar uma função de ativação como o softmax.        
            # Retorno de model quando ´last_hidden_state=True´ é setado:    
            # last_hidden_state = outputs[0], pooler_output = outputs[1], hidden_states = outputs[2]
            outputs = model(d_input_ids,
                        token_type_ids=None, 
                        attention_mask=d_input_mask, 
                        labels=d_labels)
        
        # A perda(loss) é retornado em outputs[0] porque fornecemos rótulos(labels). 
        # É útil para comparar com a perda do treinamento, quando é realizado a avaliação entre as épocas de treinamento.
        loss = outputs[0]

        # E outputs[1] os "logits" - o modelo de saídas antes da ativação.
        # logits possui duas dimensões, a primeira do lote e a segunda do rótulo da predição                        
        logits = outputs[1]
        
        # Acumule a perda da avaliação em todos os lotes para que possamos
        # calcular a perda média no final. `loss` é um tensor contendo um único valor.
        # A função '.cpu()' move loss para a cpu.
        # A função `.item ()` retorna apenas o valor Python do tensor.         
        test_losses.append(loss.cpu().item())

        # Recupera o índice do melhor resultado, maior valor dos tensores para coluna(1)
        _, classificacao = torch.max(logits, 1)

        # Verifica a classificação realizada e o rótulo previsto
        vp.append(((classificacao==1) & (d_labels==1)).sum().cpu().item())
        vn.append(((classificacao==0) & (d_labels==0)).sum().cpu().item())
        fp.append(((classificacao==1) & (d_labels==0)).sum().cpu().item())
        fn.append(((classificacao==0) & (d_labels==1)).sum().cpu().item())

        # Adiciona o documento de teste, o rótulo e a classificação realizada a lista de resultado
        for lote in range(len(d_labels)):
                    
            lista_resultado_avaliacao.append([d_documentoids[lote],
                                    d_labels[lote].cpu().item(), 
                                    classificacao[lote].cpu().item()])

        del outputs

    # Soma as classificações realizadas
    vp_s, vn_s, fp_s, fn_s = sum(vp), sum(vn), sum(fp), sum(fn)
  
    # Acurácia indica uma performance geral do modelo. 
    # Dentre todas as classificações, quantas o modelo classificou corretamente(vp=1 e vn=0).
    acc = (vp_s+vn_s)/(vp_s+vn_s+fp_s+fn_s)

    # Recall(Revocação) avalia todas as situações da classe Positivo(vp=1) com o valor esperado e quantas estão corretas.
    if (vp_s+fn_s) != 0:
        rec = (vp_s)/(vp_s+fn_s)
    else:
        rec = 0
  
    # Precisão avalia as classificações da classe positivo(vp=1 e fp=0) que o modelo fez e quantas estão corretas.
    if (vp_s+fp_s) != 0:
        pre = (vp_s)/(vp_s+fp_s)
    else:
        pre = 0  

    # F1 é a média harmônica entre precisão e recall.
    if (pre + rec) != 0:  
        f1 = 2 * ((pre * rec)/(pre + rec))
    else:
        f1 = 0

    # Média da perda da avaliação  
    media_test_loss = np.mean(test_losses)

    if model_args.use_wandb:
        # Log do wandb
        wandb.log({"acuracia": acc})
        wandb.log({"vp": vp_s})
        wandb.log({"vn": vn_s})
        wandb.log({"fp": fp_s})
        wandb.log({"fn": fn_s})
        wandb.log({"media_test_loss": media_test_loss})

    del py_input_ids
    del py_attention_masks
    del py_labels
    del test_losses
    del lote_teste_bar
    
    return media_test_loss, acc, rec, pre, f1, vp_s, vn_s, fp_s, fn_s, lista_resultado_avaliacao

# ============================
def realizaTreinamento(model_args, training_args, model, documentos_treino, classes_treino, documentoids_treino, wandb):
                       
    #training_args.num_train_epochs
    logging.info("Realizando Treinamento fold: {}".format(model_args.fold))

    # Carrega o otimizador
    otimizador = carregaOtimizador(training_args)

    # Carrega o agendador
    agendador = carregaAgendador(training_args, otimizador)

    # Defina o valor da semente em todos os lugares para torná-lo reproduzível.
    seed_val = training_args.seed

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Atualize todos os lotes ʻintervalo_atualizacao`.
    intervalo_atualizacao = obter_intervalo_atualizacao(total_iteracoes=len(documentos_treino), numero_atualizacoes=10)

    # Medida do tempo total de treinamento.
    treinamento_t0 = time.time()

    # Limpa o cache da GPU.
    torch.cuda.empty_cache()

    # Coloque o modelo em modo de treinamento. 
    model.train()

    # Acumula as perdas do treinamento.
    train_losses = []

    if model_args.use_wandb:
        # Log das métricas com wandb.
        wandb.watch(model)

    # Barra de progresso da época.
    epoca_bar = tqdm_notebook(range(training_args.num_train_epochs), desc=f'Épocas', unit=f'épocas')

    # Para cada época.
    for epoca_i in epoca_bar:
    
        # ========================================
        #               Treinamento
        # ========================================
    
        # Execute uma passada completa sobre o conjunto de treinamento.

        # Recupera o lote inteligente
        (py_input_ids, py_attention_masks, py_labels, documentoids) = cria_lotes_inteligentes(model_args, tokenizer, documentos_treino, classes_treino, documentoids_treino, training_args.per_device_train_batch_size)

        # Medida de quanto tempo leva o período de treinamento.
        treinamento_epoca_t0 = time.time()

        # Acumula as perdas do treinamento da época.
        train_epoca_losses = []

        # Barras de progresso.    
        lote_treino_bar = tqdm_notebook(range(0, len(py_input_ids)), desc=f'Epoca {epoca_i+1}', unit=f'lotes', total=len(py_input_ids) )

        # Para cada lote dos dados de treinamento.
        for index in lote_treino_bar:      

            # Progresso é atualizado a cada lotes, por exemplo, 100 lotes.
            if index % intervalo_atualizacao == 0 and not index == 0:            
                # Calcula gasto o tempo em minutos.
                tempoGasto = formataTempo(time.time() - treinamento_epoca_t0)
                        
            # Calcule o tempo restante com base em nosso progresso.
            passos_por_segundo = (time.time() - treinamento_epoca_t0) / index
            segundos_restantes = passos_por_segundo * (len(py_input_ids) - index)
            tempoRestante = formataTempo(segundos_restantes)

            # Mostra o progresso.
            logging.info("  Lote {:>7,}  de  {:>7,}.    Gasto: {:}.  Restante: {:}".format(index, len(py_input_ids), tempoGasto, tempoRestante))

            # Descompacte este lote de treinamento de nosso dataloader.
            #
            # À medida que descompactamos o lote, também copiaremos cada tensor para a GPU usando o
            # o método `to`
            #
            # `lote` é uma lista contém três tensores pytorch:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 

            # Recupera os tensores do lote e copia para a GPU usando o método `to` 
            d_input_ids = py_input_ids[index].to(device)
            d_input_mask = py_attention_masks[index].to(device)
            d_labels = py_labels[index].to(device)     
        
            # Sempre limpe quaisquer gradientes calculados anteriormente antes de realizar um
            # passe para trás. PyTorch não faz isso automaticamente porque
            # acumular os gradientes é "conveniente durante o treinamento de RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Execute um passe para frente (avalie o modelo neste lote de treinamento).
            # A documentação para esta função `model` está aqui:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Ele retorna diferentes números de parâmetros dependendo de quais argumentos
            # são fornecidos e quais sinalizadores estão definidos. Para nosso uso aqui, ele retorna
            # a perda (porque fornecemos rótulos) e os "logits" - o modelo de saídas antes da ativação.     

            # last_hidden_state = outputs[0], pooler_output = outputs[1], hidden_states = outputs[2]
            outputs = model(d_input_ids, 
                        token_type_ids=None, 
                        attention_mask=d_input_mask, 
                        labels=d_labels)
        
            # A perda(loss) é retornado em outputs[0] porque fornecemos rótulos(labels))                  
            loss = outputs[0]

            # E outputs[1] os "logits" - o modelo de saídas antes da ativação.
            # logits possui duas dimensões, a primeira do lote e a segunda do rótulo da predição                        
            # A função `.detach().cpu()` retira da gpu.
            logits = outputs[1].detach().cpu()
  
            # Acumule a perda de treinamento em todos os lotes da época para que possamos
            # calcular a perda média no final da época. `loss` é um tensor contendo um único valor.   
            # A função `.item ()` retorna apenas o valor Python do tensor.
            train_epoca_losses.append(loss.item())

            # Mostra a perda na barra de progresso.
            lote_treino_bar.set_postfix(loss=loss.item())

            if model_args.use_wandb:
                wandb.log({"train_batch_loss": loss.item()})

            # Execute uma passagem para trás para calcular os gradientes.
            # Todos os parâmetros do modelo deve ter sido setado para param.requires_grad = False
            loss.backward()            

            # Corte a norma dos gradientes para 1.0.
            # Isso ajuda a evitar o problema de "gradientes explosivos".
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       
            # Atualize os parâmetros e dê um passo usando o gradiente calculado.
            # O otimizador dita a "regra de atualização" - como os parâmetros são
            # modificados com base em seus gradientes, taxa de aprendizagem, etc.
            otimizador.step()
                           
            # Atualize a taxa de aprendizagem.
            agendador.step()

            del outputs

        # Média da perda do treinamento de todos os lotes da época.
        media_train_epoca_loss = np.mean(train_epoca_losses)

        # Acumule a perda de treinamento de todas as épocas para calcular a perda média do treinamento.    
        train_losses.append(media_train_epoca_loss)

        if model_args.use_wandb:
            wandb.log({"media_train_epoca_loss": media_train_epoca_loss})           
        
        # Medida de quanto tempo levou essa época.
        treinamento_epoca_total = formataTempo(time.time() - treinamento_epoca_t0)

        logging.info("  Média perda(loss) do treinamento da época : {0:.8f}".format(media_train_epoca_loss))
        logging.info("  Tempo de treinamento da época             : {:}".format(treinamento_epoca_total))    
        logging.info("  Tempo parcial do treinamento              : {:} (h:mm:ss)".format(formataTempo(time.time()-treinamento_t0)))

        del py_input_ids
        del py_attention_masks
        del py_labels
        del train_epoca_losses
        del lote_treino_bar    
  
    # Média da perda do treinamento de todas as épocas.
    media_train_loss = np.mean(train_losses)

    if model_args.use_wandb:
        wandb.log({"media_train_loss": media_train_loss})   

    logging.info("  Média perda(loss) treinamento : {0:.8f}".format(media_train_loss))

    del train_losses
    del epoca_bar
    del otimizador
    del agendador

    logging.info("Treinamento completo!") 
    
    return model
