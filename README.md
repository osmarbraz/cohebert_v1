# Cohebert 1.0 - Classificação e mensuração de coerência textual usando BERT
Classificação e mensuração de coerência textual utilizando o MCL BERT.


## **Instalação**

**Requisitos**

* Python 3.6.9+
* Transformer Huggingface 4.5.1
* PyTorch 1.8.1
* Spacy 2.3.5
* Wandb

**Download - Clone**

```
!git clone https://github.com/osmarbraz/cohebert_v1.git
```

**Utilização da biblioteca de funções**

Mudar o diretório corrente para a pasta clonada
```python
import sys

sys.path.append('./coebert_v1/coebert')
```

## Exemplo simples

[Este exemplo simples](notebooks/ExemploCoherenceBERT.ipynb) mostra como utilizar o CoheBERT e um modelo BERT pré-treinado para medir a coerência de documentos.

Depois de instalar as bibliotecas necessárias podemos fazer uso da biblioteca CoheBERT.

Realizamos o import das bibliotecas e instanciamos a classe CoherenceBERT para que seja realizado download do modelo pré-treinado.

````python
from coherence_bert import CoherenceBERT

cohebert = CoherenceBERT('neuralmind/bert-large-portuguese-cased') # BERTimbau large
````

Em seguida, forneça algumas sentenças ao CoheBERT e recupere a medida.

````python
DO = ['Bom Dia, professor.',
      'Qual o conteúdo da prova?',
      'Vai cair tudo na prova?',
      'Aguardo uma resposta, João.']      

# Recupera as medidas dos documentos
CcosDO =  cohebert.getMedidaCoerenciaCosseno(DO)

print('Ccos DO    :', CcosDO) #Ccos DO1    : 0.8178287347157797
````

## Medidas de (In)coerência
Para obter a medida de coerência das sentenças utilize as operações a seguir passando o texto como parâmetro:

* ```getMedidaCoerenciaCosseno``` - Retorna a medida de coerência utilizando a similaridade cosseno das sentenças.
* ```getMedidaCoerenciaEuclidiana``` - Retorna a medida de coerência utilizando a distância de Euclidiana das sentenças.
* ```getMedidaCoerenciaManhattan``` - Retorna a medida de coerência utilizando a distância de Manhattan das sentenças.

## Modelos Pré-treinados do BERT

Apesar de existir uma lista grande de [Modelos Pré-treinados](https://huggingface.co/models) testamos somente três modelos: 
* ```neuralmind/bert-base-portuguese-cased``` - BERTimbau base
* ```neuralmind/bert-large-portuguese-cased``` - BERTimbau large
* ```bert-base-multilingual-cased``` - BERT Multilingual


## Exemplos completos

Exemplos de execução completos com o CoheBERT são mostrados nos [notebooks](notebooks/cstnews/).

## **Diretórios**
* **cohebert** - Código fonte do CoheBERT versão 1.0
* **conjuntodedados** - Diretório com os conjuntos de dados.
  * **cstnews** - Arquivos do conjunto de dados do CSTNews.
  * **onlineeduc1.0** - Arquivos do conjunto de dados do OnlineEduc 1.0 (**Não disponibilizado**).

* **experimentos** - Diretório com o resultados dos experimentos.
  * **cstnews** - Arquivos dos resultados do CSTNews.
    * **validacao_classificacao** - Resultados dos experimentos de classificação.
    * **validacao_medicao** - Resultados dos experimentos de cálculo de medida.
  * **onlineeduc1.0** - Arquivos dos resultados do OnlineEduc 1.0.
    * **validacao_classificacao** - Resultados dos experimentos de classificação.
    * **validacao_medicao** - Resultados dos experimentos de cálculo de medida.

* **notebooks** - Diretório com o notebooks dos experimentos.
  * **cstnews** - Notebooks para classificação e mensuração da coerência no conjunto de dados CSTNews. Utiliza o BERTimbau e o BERTMultilingual para classificar e medir a coerência de textos. 
    * **GerarDadosValidacaoKfold_CSTNews_v1.ipynb** - Gera os dados para validação cruzada do conjunto de dados.
    * **AnaliseDadosCSTNews_v1.ipynb** - Realiza a análise do conjunto de dados.
    * **MedidaCoerenciaCSTNews_v1_pretreinado.ipynb** - Realiza os cálculos de medida de (in)coerência.
    * **AjusteFinoCSTNews_v1_C_SB_KFold_Todos.ipynb** - Realiza o ajuste fino do MCL BERT Pré-treinado para todos os folds de uma parâmetrização.
    * **AjusteFinoCSTNews_AvaliacaoOnlineeduc1.0_v1_C_SB.ipynb** - Realiza o ajuste fino do MCL BERT Pré-treinado com os dados do CSTNews e a avaliação com os dados do OnlineEduc 1.0.
    * **CorrelacaoCSTNews_ClassificadorCalculo_v1.ipynb** - Realiza o análise dos resultados do classificador e o cálculo das medidas de coerência.
 
