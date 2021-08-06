# CoheBERT 1.0 - Classificação e mensuração de coerência textual usando BERT
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

sys.path.append("./coebert_v1/coebert")
```

## Exemplo simples

[Este exemplo simples](notebooks/ExemploCoherenceBERT.ipynb) mostra como utilizar o **CoheBERT** e um modelo BERT pré-treinado para medir a coerência de documentos.

Depois de instalar as bibliotecas necessárias podemos fazer uso da biblioteca **CoheBERT**.

Realizamos o import das bibliotecas do **CoheBERT** e instanciamos a classe CoherenceBERT para que seja realizado download do modelo pré-treinado. O download é realizado da comunidade ou de uma url.

````python
from coherence_bert import CoherenceBERT

cohebert = CoherenceBERT("neuralmind/bert-large-portuguese-cased") # BERTimbau large
````

Em seguida, forneça um documento com suas sentenças separadas em uma lista ao **CoheBERT** e recupere a medida.

````python
# Documento e suas sentenças
DO = ["Bom Dia, professor.",
      "Qual o conteúdo da prova?",
      "Vai cair tudo na prova?",
      "Aguardo uma resposta, João."]

# Recupera a medida Ccos do documento
CcosDO = cohebert.getMedidaCoerenciaCosseno(DO)

# Mostra a medida recuperada
print("Ccos DO    :", CcosDO) #Ccos DO    : 0.8178287347157797
````

Este exemplo permuta as sentenças antes de submeter ao **CoheBERT**.

````python
# Documento e suas sentenças permutadas
permDO = ["Aguardo uma resposta, João.",
          "Qual o conteúdo da prova?",              
          "Bom Dia, professor.",
          "Vai cair tudo na prova?"]         

# Recupera a medida Ccos do documento
CcospermDO = cohebert.getMedidaCoerenciaCosseno(permDO)

# Mostra a medida recuperada
print("Ccos permDO    :", CcospermDO) #Ccos permDO    : 0.7760167121887207
````

## Medidas de (In)coerência **CoheBERT**
Para obter a medida de coerência das sentenças utilize as operações a seguir passando o texto como parâmetro:

* ```getMedidaCoerenciaCosseno``` - Retorna a medida de coerência utilizando a similaridade cosseno das sentenças.
* ```getMedidaCoerenciaEuclidiana``` - Retorna a medida de coerência utilizando a distância de Euclidiana das sentenças.
* ```getMedidaCoerenciaManhattan``` - Retorna a medida de coerência utilizando a distância de Manhattan das sentenças.

## Modelos Pré-treinados do BERT

Apesar de existir uma lista grande de [Modelos Pré-treinados](https://huggingface.co/models) testamos no **CoheBERT** somente três modelos pré-treinados para a língua portuguesa: 
* ```neuralmind/bert-base-portuguese-cased``` - [BERTimbau base](https://github.com/neuralmind-ai/portuguese-bert)
* ```neuralmind/bert-large-portuguese-cased``` - [BERTimbau large](https://github.com/neuralmind-ai/portuguese-bert)
* ```bert-base-multilingual-cased``` - BERT Multilingual


## Exemplos completos

Exemplos de execução completos com o **CoheBERT** são mostrados nos [notebooks](notebooks/cstnews/).

## **Diretórios**

Relação e descrição dos principais diretórios do cohebert.

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
    * **AjusteFinoCSTNews_AvaliacaoOnlineEduc_v1_C_SB.ipynb** - Realiza o ajuste fino do MCL BERT Pré-treinado com os dados do CSTNews e a avaliação com os dados do OnlineEduc 1.0.
    * **CorrelacaoCSTNews_ClassificadorCalculo_v1.ipynb** - Realiza o análise dos resultados do classificador e o cálculo das medidas de coerência.
 

## Citando & Autores
Se achar este repositório útil, sinta-se à vontade para citar nossa publicação [CoheBERT: Coherence BERT](https://):

```bibtex 
@inproceedings{brazfileto-2021-cohebert,
    title = "xxxx",
    author = "Braz, Osmar and Fileto, Renato",
    booktitle = "xxxxxx",
    year = "2021",
    publisher = "xxx",
    url = "https://xxxxx",
}
```
