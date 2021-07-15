# Import das bibliotecas.
from dataclasses import dataclass, field
from typing import Dict, Optional
from typing import List

@dataclass
class ModeloArgumentosMedida:
        
    '''
    Classe(ModeloArgumentosMedida) de definição dos parâmetros do modelo para o cálculo de medidas.
    '''
 
    max_seq_len: Optional[int] = field(
        default=None,
        metadata={'help': 'max seq len'},
    )    
    pretrained_model_name_or_path: str = field(
        default='neuralmind/bert-base-portuguese-cased',
        metadata={'help': 'nome do modelo pré-treinado do BERT.'},
    )
    modelo_spacy: str = field(
        default='pt_core_news_lg',
        metadata={'help': 'nome do modelo do spaCy.'},
    )
    versao_spacy: str = field(
        default='2.3.0',
        metadata={'help': 'versão do spaCy a ser utilizada.'},
    )   
    do_lower_case: bool = field(
        default=False,
        metadata={'help': 'define se o texto do modelo deve ser todo em minúsculo.'},
    )  
    output_attentions: bool = field(
        default=False,
        metadata={'help': 'habilita se o modelo retorna os pesos de atenção.'},
    )
    output_hidden_states: bool = field(
        default=False,
        metadata={'help': 'habilita gerar as camadas ocultas do modelo.'},
    )
    use_wandb : bool = field(
        default=True,
        metadata={'help': 'habilita o uso do wandb.'},
    )
    salvar_modelo_wandb : bool = field(
        default=True,
        metadata={'help': 'habilita o salvamento do modelo no wandb.'},
    )
    salvar_avaliacao : bool = field(
        default=True,
        metadata={'help': 'habilita o salvamento do resultado da avaliação.'},
    )     
    salvar_medicao : bool = field(
        default=False,
        metadata={'help': 'habilita o salvamento da medicao.'},
    )
    usar_mcl_ajustado : bool = field(
        default=False,
        metadata={'help': 'habilita o carragamento de mcl ajustado.'},
    )
    estrategia_pooling: int = field(
        default=0, # 0 - MEAN estratégia de pooling média / 1 - MAX  estratégia de pooling maior
        metadata={'help': 'Estratégia de pooling de padronização do embeddings das= palavras das sentenças.'},
    )
    palavra_relevante: int = field(
        default=0, # 0 - ALL Considera todas as palavras das sentenças / 1 - CLEAN desconsidera as stopwords / 2 - NOUN considera somente as palavras substantivas
        metadata={'help': 'Estratégia de relevância das palavras das sentenças para gerar os embeddings.'},
    )

        
@dataclass
class ModeloArgumentosClassificacao:
        
    '''
    Classe(ModeloArgumentosClassificacao) de definição dos parâmetros do modelo para a classificação.
    '''
 
    max_seq_len: Optional[int] = field(
        default=None,
        metadata={'help': 'max seq len'},
    )    
    pretrained_model_name_or_path: str = field(
        default='neuralmind/bert-base-portuguese-cased',
        metadata={'help': 'nome do modelo pré-treinado do BERT.'},
    )   
    do_lower_case: bool = field(
        default=False,
        metadata={'help': 'define se o texto do modelo deve ser todo em minúsculo.'},
    ) 
    num_labels: int = field(
        default=2,
        metadata={"help": "número de rótulos a ser classificado."},
    )
    output_attentions: bool = field(
        default=False,
        metadata={'help': 'habilita se o modelo retorna os pesos de atenção.'},
    )
    output_hidden_states: bool = field(
        default=False,
        metadata={'help': 'habilita gerar as camadas ocultas do modelo.'},
    )
    use_wandb : bool = field(
        default=True,
        metadata={'help': 'habilita o uso do wandb.'},
    )
    salvar_modelo_wandb : bool = field(
        default=True,
        metadata={'help': 'habilita o salvamento do modelo no wandb.'},
    )
   salvar_classificacao : bool = field(
        default=False,
        metadata={"help": "habilita o salvamento da classificação."},
    )
    salvar_avaliacao : bool = field(
        default=True,
        metadata={"help": "habilita o salvamento do resultado da avaliação."},
    )  
    usar_mcl_ajustado : bool = field(
        default=False,
        metadata={'help': 'habilita o carragamento de mcl ajustado.'},
    )   
    fold: int = field(
        default="1",
        metadata={"help": "quantidade de folds a serem gerados"},
    ) 
