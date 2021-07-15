# Import das bibliotecas.
import requests # Biblioteca para download
import zipfile # Biblioteca para descompactar
import os # Biblioteca para apagar arquivos
import shutil # Biblioteca para mover arquivos    
from transformers import BertModel # Importando as bibliotecas do Modelo BERT   
from transformers import BertTokenizer # Importando as bibliotecas do tokenizador BERT

def getNomeModeloBERT(model_args):
    '''    
    Recupera uma string com uma descrição do modelo BERT para nomes de arquivos e diretórios.
    '''

    # Verifica o nome do modelo(default SEM_MODELO_BERT)
    MODELO_BERT = 'SEM_MODELO_BERT'
    if 'neuralmind' in model_args.pretrained_model_name_or_path:
        MODELO_BERT = '_BERTimbau'
    else:
        if 'multilingual' in model_args.pretrained_model_name_or_path:
            MODELO_BERT = '_BERTmultilingual'
            
    return MODELO_BERT

def getTamanhoBERT(model_args):
    '''    
    Recupera uma string com o tamanho(dimensão) do modelo BERT para nomes de arquivos e diretórios.
    '''
    
    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = '_large'
    if 'base' in model_args.pretrained_model_name_or_path:
        TAMANHO_BERT = '_base'
        
    return TAMANHO_BERT  

def downloadModeloPretreinado(model_args):
    ''' 
    Realiza o download do modelo BERT(MODELO) e retorna o diretório onde o modelo BERT(MODELO) foi descompactado.
    ''' 

    MODELO = model_args.pretrained_model_name_or_path

    # Variável para setar o arquivo.
    URL_MODELO = None

    if 'http' in MODELO:
        URL_MODELO = MODELO

    # Se a variável foi setada.
    if URL_MODELO:

        # Diretório descompactação.
        DIRETORIO_MODELO = '/content/modelo'

        # Recupera o nome do arquivo do modelo da url.
        ARQUIVO = URL_MODELO.split('/')[-1]

        # Nome do arquivo do vocabulário.
        ARQUIVO_VOCAB = 'vocab.txt'

        # Caminho do arquivo na url.
        CAMINHO_ARQUIVO = URL_MODELO[0:len(URL_MODELO)-len(ARQUIVO)]

        # Verifica se a pasta de descompactação existe na pasta corrente
        if os.path.exists(DIRETORIO_MODELO):
            print('Apagando diretório existente do modelo!')
            # Apaga a pasta e os arquivos existentes                     
            shutil.rmtree(DIRETORIO_MODELO)

        # Realiza o download do arquivo do modelo        
        data = requests.get(URL_MODELO)
        arquivo = open(ARQUIVO, 'wb')
        arquivo.write(data.content)

        # Descompacta o arquivo na pasta de descompactação.                
        arquivoZip = zipfile.ZipFile(ARQUIVO,"r")
        arquivoZip.extractall(DIRETORIO_MODELO)

        # Baixa o arquivo do vocabulário.
        # O vocabulário não está no arquivo compactado acima, mesma url mas arquivo diferente.
        URL_MODELO_VOCAB = CAMINHO_ARQUIVO + ARQUIVO_VOCAB        
        # Coloca o arquivo do vocabulário no diretório de descompactação.
        data = requests.get(URL_MODELO_VOCAB)
        arquivo = open(DIRETORIO_MODELO + "/" + ARQUIVO_VOCAB, 'wb')
        arquivo.write(data.content)

        # Apaga o arquivo compactado
        os.remove(ARQUIVO)

        print('Pasta do {} pronta!'.format(DIRETORIO_MODELO))

    else:
        DIRETORIO_MODELO = MODELO
        print('Variável URL_MODELO não setada!')

    return DIRETORIO_MODELO

def copiaModeloAjustado():
    ''' 
    Copia o MODELO ajustado do GoogleDrive para o projeto.
    ''' 

    # Diretório local de salvamento do modelo.
    DIRETORIO_LOCAL_MODELO_AJUSTADO = '/content/modelo_ajustado/'

    # Diretório remoto de salvamento do modelo.
    DIRETORIO_REMOTO_MODELO_AJUSTADO = '/content/drive/MyDrive/Colab Notebooks/Data/CSTNEWS/validacao_classificacao/holdout/modelo/modelo' + MODELO_BERT + TAMANHO_BERT

    # Copia o arquivo do modelo para o diretório no Google Drive.
    shutil.copytree(DIRETORIO_REMOTO_MODELO_AJUSTADO, DIRETORIO_LOCAL_MODELO_AJUSTADO) 
   
    print('Modelo copiado!')

    return DIRETORIO_LOCAL_MODELO_AJUSTADO

def verificaModelo(model_args):
    ''' 
    Verifica de onde utilizar o modelo
    ''' 

    DIRETORIO_MODELO = None
    
    if model_args.usar_mcl_ajustado == True:
        DIRETORIO_MODELO = copiaModeloAjustado()
        print('Usando modelo ajustado')
        
    else:
        DIRETORIO_MODELO = downloadModeloPretreinado(model_args)
        print('Usando modelo pré-treinado de download ou comunidade')
        
    return DIRETORIO_MODELO

def carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o tokenizador do MODELO.
    O tokenizador utiliza WordPiece.
    Carregando o tokenizador da pasta '/content/modelo/' do diretório padrão se variável `DIRETORIO_MODELO` setada.
    Caso contrário carrega da comunidade
    Por default(`do_lower_case=True`) todas as letras são colocadas para minúsculas. Para ignorar a conversão para minúsculo use o parâmetro `do_lower_case=False`. Esta opção também considera as letras acentuadas(ãçéí...), que são necessárias a língua portuguesa.
    O parâmetro `do_lower_case` interfere na quantidade tokens a ser gerado a partir de um texto. Quando igual a `False` reduz a quantidade de tokens gerados.
    ''' 

    # Se a variável DIRETORIO_MODELO foi setada.
    if DIRETORIO_MODELO:

        # Carregando o Tokenizador.
        print('Carregando o tokenizador BERT do diretório {}...'.format(DIRETORIO_MODELO))

        tokenizer = BertTokenizer.from_pretrained(DIRETORIO_MODELO, 
                                                  do_lower_case=model_args.do_lower_case)

    else:
        # Carregando o Tokenizador da comunidade.
        print('Carregando o tokenizador da comunidade...')

        tokenizer = BertTokenizer.from_pretrained(model_args.pretrained_model_name_or_path, 
                                                  do_lower_case=model_args.do_lower_case)

    return tokenizer

def carregaModelo(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o modelo e retorna o modelo.
    ''' 

    # Variável para setar o arquivo.
    URL_MODELO = None

    if 'http' in model_args.pretrained_model_name_or_path:
        URL_MODELO = model_args.pretrained_model_name_or_path

    # Se a variável URL_MODELO foi setada
    if URL_MODELO:
        # Carregando o Modelo BERT
        print('Carregando o modelo BERT do diretório {}...'.format(DIRETORIO_MODELO))

        model = BertModel.from_pretrained(DIRETORIO_MODELO,
                                          output_attentions = model_args.output_attentions,
                                          output_hidden_states = model_args.output_hidden_states)
    else:
        # Carregando o Modelo BERT da comunidade
        print('Carregando o modelo BERT da comunidade ...')

        model = BertModel.from_pretrained(model_args.pretrained_model_name_or_path,
                                          output_attentions = model_args.output_attentions,
                                          output_hidden_states = model_args.output_hidden_states)

    return model

def carregaBERT(model_args):
    ''' 
    Carrega o BERT e retorna o modelo e o tokenizador.
    ''' 
    
    # Verifica a origem do modelo
    DIRETORIO_MODELO = verificaModelo(model_args)
    
    # Carrega o modelo
    model = carregaModelo(DIRETORIO_MODELO, model_args)
    
    # Carrega o tokenizador
    tokenizer = carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args)
    
    return model, tokenizer
