# Import das bibliotecas.
import logging  # Biblioteca de logging
from transformers import AdamW # Biblioteca do otimizador
from transformers import get_linear_schedule_with_warmup # Biblioteca do agendador

# ============================
def carregaOtimizador(training_args):
    '''
    Esta função carrega o otimizador utilizado no agendador de aprendizado.
    Parâmetros:
        `training_args` - Objeto com os argumentos do treinamento. 
    '''
    # Nota: AdamW é uma classe da biblioteca huggingface (ao contrário de pytorch).
    # Eu acredito que o 'W' significa 'Correção de redução de peso "
    optimizer = AdamW(model.parameters(),
                  lr = training_args.learning_rate, # (ou alfa) A taxa de aprendizado a ser usada. - default é 3e-5
                  # betas = (0.9, 0.999), # (beta1, beta2) - default é (0.9, 0.999)
                    # beta1 é taxa de decaimento exponencial para as estimativas do primeiro momento. 
                    # beta2 é taxa de decaimento exponencial para as estimativas do segundo momento. Este valor deve ser definido próximo a 1,0 em problemas com gradiente esparso (por exemplo, PNL e problemas de visão de computacional)
                  # eps = 1e-6, #  É um número muito pequeno para evitar qualquer divisão por zero na implementação - default é 1e-6.
                  # weight_decay = 0.0, # Correção de redução de peso. - default é 0.0
                    # A redução da taxa de aprendizagem também pode ser usada com Adam. A taxa de decaimento é atualizada a cada época para a demonstração da regressão logística.
                  # correct_bias = True #  Se não deve corrigir o viés(bias) no Adam mudar para False.- default é True
                )
  
    return optimizer

  
  
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
