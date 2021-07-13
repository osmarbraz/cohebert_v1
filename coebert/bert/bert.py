def getNomeModeloBERT(model_args):
    '''    
    Recupera a descri��o do modelo para nomes de arquivos e diret�rios.
    '''

    # Verifica o nome do modelo(default BERT)
    MODELO_BERT = 'SEM_MODELO_BERT'
    if 'neuralmind' in model_args.pretrained_model_name_or_path:
        MODELO_BERT = '_BERTimbau'
    else:
        if 'multilingual' in model_args.pretrained_model_name_or_path:
            MODELO_BERT = '_BERTmultilingual'
    return MODELO_BERT

def getTamanhoBERT(model_args):
    '''    
    Recupera a dimens�o do modelo para nomes de arquivos e diret�rios.
    '''
    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = '_large'
    if 'base' in model_args.pretrained_model_name_or_path:
        TAMANHO_BERT = '_base'
    return TAMANHO_BERT  


def downloadModeloPretreinado(MODELO):
    ''' 
    Realiza o download do MODELO e retorna o diret�rio onde o MODELO foi descompactado.
    ''' 

    # Importando as bibliotecas.
    import os

    # Vari�vel para setar o arquivo.
    URL_MODELO = None

    if 'http' in MODELO:
        URL_MODELO = MODELO

    # Se a vari�vel foi setada.
    if URL_MODELO:

        # Diret�rio descompacta��o.
        DIRETORIO_MODELO = '/content/modelo'

        # Recupera o nome do arquivo do modelo da url.
        arquivo = URL_MODELO.split('/')[-1]

        # Nome do arquivo do vocabul�rio.
        arquivo_vocab = 'vocab.txt'

        # Caminho do arquivo na url.
        caminho = URL_MODELO[0:len(URL_MODELO)-len(arquivo)]

        # Verifica se a pasta de descompacta��o existe na pasta corrente
        if os.path.exists(DIRETORIO_MODELO):
            print('Apagando diret�rio existente do modelo!')
            # Apaga a pasta e os arquivos existentes
            !rm -rf $DIRETORIO_MODELO    

        # Baixa o arquivo do modelo.
        !wget $URL_MODELO

        # Descompacta o arquivo na pasta de descompacta��o.
        !unzip -o $arquivo -d $DIRETORIO_MODELO

        # Baixa o arquivo do vocabul�rio.
        # O vocabul�rio n�o est� no arquivo compactado acima, mesma url mas arquivo diferente.
        URL_MODELO_VOCAB = caminho + arquivo_vocab
        !wget $URL_MODELO_VOCAB

        # Coloca o arquivo do vocabul�rio no diret�rio de descompacta��o.
        !mv $arquivo_vocab $DIRETORIO_MODELO

        # Move o arquivo para pasta de descompacta��o
        !mv $arquivo $DIRETORIO_MODELO

        print('Pasta do {} pronta!'.format(DIRETORIO_MODELO))

        # Lista a pasta corrente.
        !ls -la $DIRETORIO_MODELO
    else:
        DIRETORIO_MODELO = MODELO
        print('Vari�vel URL_MODELO n�o setada!')

    return DIRETORIO_MODELO

def copiaModeloAjustado():
    ''' 
    Copia o MODELO ajustado do GoogleDrive para o projeto.
    ''' 

    # Importando as bibliotecas.
    import os

    # Diret�rio local de salvamento do modelo.
    DIRETORIO_LOCAL_MODELO_AJUSTADO = '/content/modelo_ajustado/'

    # Diret�rio remoto de salvamento do modelo.
    DIRETORIO_REMOTO_MODELO_AJUSTADO = '/content/drive/MyDrive/Colab Notebooks/Data/CSTNEWS/validacao_classificacao/holdout/modelo/modelo' + MODELO_BERT + TAMANHO_BERT

    ## Copia o arquivo do modelo para o diret�rio no Google Drive.
    !cp -r '$DIRETORIO_REMOTO_MODELO_AJUSTADO' '$DIRETORIO_LOCAL_MODELO_AJUSTADO' 

    print('Modelo copiado!')

    return DIRETORIO_LOCAL_MODELO_AJUSTADO

def verificaModelo():
    ''' 
    Verifica de onde utilizar o modelo
    ''' 

    DIRETORIO_MODELO = None
    if model_args.usar_mcl_ajustado == True:
        DIRETORIO_MODELO = copiaModeloAjustado()
        print('Usando modelo ajustado')
    else:
        DIRETORIO_MODELO = downloadModeloPretreinado(model_args.pretrained_model_name_or_path)
        print('Usando modelo pr�-treinado de download ou comunidade')
    return DIRETORIO_MODELO

def carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO):
    ''' 
    Carrega o tokenizador do MODELO.
    O tokenizador utiliza WordPiece.
    Carregando o tokenizador da pasta '/content/modelo/' do diret�rio padr�o se vari�vel `DIRETORIO_MODELO` setada.
    Caso contr�rio carrega da comunidade
    Por default(`do_lower_case=True`) todas as letras s�o colocadas para min�sculas. Para ignorar a convers�o para min�sculo use o par�metro `do_lower_case=False`. Esta op��o tamb�m considera as letras acentuadas(����...), que s�o necess�rias a l�ngua portuguesa.
    O par�metro `do_lower_case` interfere na quantidade tokens a ser gerado a partir de um texto. Quando igual a `False` reduz a quantidade de tokens gerados.
    ''' 

    # Importando as bibliotecas do tokenizador.
    from transformers import BertTokenizer

    # Se a vari�vel DIRETORIO_MODELO foi setada.
    if DIRETORIO_MODELO:

        # Carregando o Tokenizador.
        print('Carregando o tokenizador BERT do diret�rio {}...'.format(DIRETORIO_MODELO))

        tokenizer = BertTokenizer.from_pretrained(DIRETORIO_MODELO, 
                                                  do_lower_case=model_args.do_lower_case)

    else:
        # Carregando o Tokenizador da comunidade.
        print('Carregando o tokenizador da comunidade...')

        tokenizer = BertTokenizer.from_pretrained(model_args.pretrained_model_name_or_path, 
                                                  do_lower_case=model_args.do_lower_case)

    return tokenizer


def carregaModelo(MODELO, DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o modelo e retorna o modelo.
    ...

    # Importando as bibliotecas do Modelo
    from transformers import BertModel

    # Vari�vel para setar o arquivo.
    URL_MODELO = None

    if 'http' in MODELO:
        URL_MODELO = MODELO

    # Se a vari�vel URL_MODELO foi setada
    if URL_MODELO:
        # Carregando o Modelo BERT
        print('Carregando o modelo BERT do diret�rio {}...'.format(DIRETORIO_MODELO))

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

