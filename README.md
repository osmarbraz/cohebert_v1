# Investigando coerência em postagens de um fórum de dúvidas em ambiente virtual de aprendizagem com o BERT

## **Resumo**
Incoerências podem causar dificuldades na interpretação de discursos e impactar o desempenho de agentes conversacionais e tutores inteligentes, entre outros. Modelos contextualizados de linguagem como o BERT não foram ainda explorados na análise de incoerência, a despeito de sua eficácia comprovada em diversas tarefas afins. Este trabalho usa variações do BERT em língua portuguesa para classificar e medir coerência textual. Experimentos com textos de notícias e de um fórum educacional de dúvidas de estudantes mostram que o BERT suporta discriminação da ordem de sentenças com até 99,20% de acurácia e algumas medidas de (in)coerência consistentes com tal classificação, sendo a maioria dos melhores resultados para os textos do fórum. 

## **Instalação**

**Requisitos**

* Python 3.6.9+
* [Transformer Huggingface 4.5.1](https://huggingface.co/transformers/)
* [PyTorch 1.8.1](https://pytorch.org/)
* [Spacy 2.3.5](https://spacy.io/)

**Download - Clone**

```
!git clone https://github.com/osmarbraz/cohebert_v1.git
```

**Utilização da biblioteca de funções**

Mudar o diretório corrente para a pasta clonada
```python
import sys

sys.path.append("./cohebert_v1/cohebert")
```

## Exemplo simples

[Este exemplo simples](notebooks/ExemploCoheBERT.ipynb) mostra como o pacote **cohebert_v1** e um modelo BERT pré-treinado são usados para medir a coerência de documentos através da permutação de sentenças.

Depois de instalar as bibliotecas necessárias podemos fazer uso da biblioteca **cohebert_v1**.

Realizamos o import das bibliotecas de **cohebert_v1** e instanciamos um objeto da classe **CoheBERT** para que seja realizado download do modelo pré-treinado. O download é realizado da comunidade ou de uma url.

````python
from cohebert import CoheBERT

cohebert = CoheBERT("neuralmind/bert-large-portuguese-cased") # BERTimbau large
````

Em seguida, forneça algumas sentenças separadas em uma lista para o **CoheBERT** calcular a coerência dessas sentenças usando alguma função de similaridade ou distância, tais como a similaridade cosseno ou a distância Euclidiana ou Manhathan.

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

## Medidas de (In)coerência disponíveis
Para obter a medida de (in)coerência das sentenças utilize as operações a seguir passando o texto como parâmetro:

* ```getMedidaCoerenciaCosseno``` - Retorna a medida de coerência utilizando a similaridade cosseno das sentenças. 
* ```getMedidaCoerenciaEuclidiana``` - Retorna a medida de incoerência utilizando a distância de Euclidiana das sentenças.
* ```getMedidaCoerenciaManhattan``` - Retorna a medida de incoerência utilizando a distância de Manhattan das sentenças.

## Modelos Pré-treinados do BERT

Apesar de existir uma lista grande de [Modelos Pré-treinados](https://huggingface.co/models) testamos no **CoheBERT** somente três modelos pré-treinados para a língua portuguesa: 
* ```neuralmind/bert-base-portuguese-cased``` - [BERTimbau base](https://github.com/neuralmind-ai/portuguese-bert)
* ```neuralmind/bert-large-portuguese-cased``` - [BERTimbau large](https://github.com/neuralmind-ai/portuguese-bert)
* ```bert-base-multilingual-cased``` - [BERT Multilingual](https://huggingface.co/bert-base-multilingual-cased)


## Exemplos completos

Exemplos de execução completos com o **CoheBERT** são mostrados nos [notebooks](notebooks/cstnews/).

## **Diretórios**

Relação e descrição dos principais diretórios do cohebert.

* **cohebert** - Código fonte do CoheBERT.
* **conjuntodedados** - Diretório com os conjuntos de dados.
  * **cstnews** - Arquivos do conjunto de dados do CSTNews.
  * **onlineeduc1.0** - Arquivos do conjunto de dados do OnlineEduc 1.0 (**Não disponibilizado** por se tratar de conjunto de dados de propriedade de uma universidade).

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
 
## Registro dos resultados dos experimentos

Os resultados dos experimentos foram armazenados na ferramenta Wandb.

*  Classificação: [CSTNews](https://wandb.ai/osmar-braz/ajustefinocstnews_v1_c_sb_kfold?workspace=user-osmar-braz) e [OnlineEduc1.0](https://wandb.ai/osmar-braz/ajustefinomoodle_v1_c_sb_kfold?workspace=user-osmar-braz)
*  Mensuração: [CSTNews](https://wandb.ai/osmar-braz/medidacoerenciamoodle_v1?workspace=user-osmar-braz) e [OnlineEduc1.0](https://wandb.ai/osmar-braz/medidacoerenciamoodle_v1?workspace=user-osmar-braz)

## Citando & Autores

Se achar este repositório útil, sinta-se à vontade para citar nossa [publicação](https://):

```bibtex 
@inproceedings{brazfileto-2021-cohebert,
    title = "Investigando coerência em postagens de um fórum de dúvidas em ambiente virtual de aprendizagem com o BERT",
    author = "Braz, Osmar and Fileto, Renato",
    booktitle = "Anais do XXXII Simpósio Brasileiro de Informática na Educação",
	pages={XXXX-XXXX},
    year = "2021",    
	organization={SBC},
    url = "https://xxxxx",
}
```
