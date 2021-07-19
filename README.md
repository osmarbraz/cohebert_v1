# coebert_v1 - Classificação e mensuração de coerência textual usando BERT
Classificação e mensuração de coerência textual utilizando o MCL BERT.


## **Instalação**

**Requisitos**

* Python 3.6.9+
* Transformer Huggingface 4.5.1
* PyTorch 1.8.1
* Spacy 2.3.5
* Wandb

**Download**

```
!git clone https://github.com/osmarbraz/coebert_v1.git
```

**Execução**

Mudar o diretório corrente para a pasta clonada
```python
import sys

sys.path.append('./coebert_v1/coebert')
```

## **Diretórios**
* **coebert** - Código fonte do coeberb v 1.0
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
    * **AnaliseDadosCSTNews_v1.ipynb** - Realiza a análise dos dados do conjunto de dados.
    * **MedidaCoerenciaCSTNews_v1_pretreinado.ipynb** - Realiza os cálculos de medida de (in)coerência.
    * **AjusteFinoCSTNews_v1_C_SB_KFold_Todos.ipynb** - Realiza o ajuste fino do MCL BERT Pré-treinado para todos os folds para uma parâmetrização.
    * **AjusteFinoCSTNews_AvaliacaoOnlineeduc1.0_v1_C_SB.ipynb** - Realiza o ajuste fino do MCL BERT Pré-treinado com o CSTNews e a avaliação com o OnlineEduc 1.0.
    * **CorrelacaoCSTNews_ClassificadorCalculo_v1.ipynb** - Realiza o análise de correção do classificador e o cálculo de medidas.
 
