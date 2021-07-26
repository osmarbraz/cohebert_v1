# Import das bibliotecas.
from transformers import AdamW


def carregaOtimizador(training_args):

  '''
    Esta função carrega o otimizador utilizado no agendador de aprendizado.
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
