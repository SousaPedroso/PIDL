# _Post-hoc_ Interpretable Deep Learning for Bioacustics data (PIDL)

Esse repositório contém os códigos para os experimentos utilizados para a construção do meu trabalho de conclusão de curso (TCC) em [Ciência da Computação](https://www.ic.ufmt.br/) pela [UFMT](https://www.ufmt.br/). A seguir mostro a tabela de seções do repositório e depois o que contém cada uma.

## Tabela de seções

- [Motivação](#motivação)
- [Estrutura](#estrutura)
- [Configuração](#configuração)

## Motivação

Ao longo de minha iniciação científica (IC) na graduação, encontrei alguns _papers_ a respeito de  **arquiteturas interpretáveis** de _DL_ (e.g. [Prototype Learning](https://ieeexplore.ieee.org/document/9747014/)) para sons bem como métodos de interpretabilidade para Redes Neurais com modelos já treinados, e.g [DeepLift](https://arxiv.org/abs/1704.02685). Para ajudar mais o grupo de pesquisa [CO.BRA](https://cobra.ic.ufmt.br/) ao qual participei com IC e entender também algumas dificuldades ao longo destes anos, decidi seguir com esse tópico de estudo, por ser algo ainda não trabalhado. Há muito espaço para melhorias com interpretabilidade não só na bioacústica, como em outras áreas, mas por ser um trabalho de conclusão de curso, me limitei a algo pouco explorado na bioacústica (métodos _post-hoc_), mas que não fosse tão simples de entender/aplicar. Dentre algumas outras opções, podem ser realizados testes com outras bases de dados bem como outros modelos para maior certificação da importância de interpretabilidade ou **possível** generalização de características aprendidas. Explico com mais detalhes em meu TCC.

Houve limitação quanto ao número de possibilidades a fazer devido ao tempo de processamento. Infelizmente não foi possível a utilização do servidor que havia o acesso anteriormente a meu TCC, mas tive que adequar com a minha máquina sem GPU. Independente de quão alavancados poderiam ser os resultados com mais processamento, a ideia ainda é válida e mostrou ser viável; com necessidade de mais testes com modelos, bases, entre outros, como elenco em trabalhos futuros.

## Estrutura

Esta seção é para detalhar como foi pensada a estrutura de diretórios. O diretório _PANNs_ consiste de um ambiente cujas dependências são controladas através do [pipenv](https://github.com/pypa/pipenv). Mudanças nos parâmetros do modelo são rastreadas através do [mlflow](https://github.com/mlflow/mlflow). Mudanças em processamentos podem ser feitas com anotações através de _tags_ (ou ainda salvando arquivos necessários, que não foi preciso aqui). Dentro de [panns](/panns/), a pasta [utils](/panns/utils/) contém códigos auxiliares necessários para todos os módulos.

## Configuração

Esta seção mostra como você pode reproduzir meus experimentos. Não somente o treinamento e validação dos modelos são reproduzíveis, mas a ideia é também permitir um ambiente reproduzível para que você precise somente fazer a instalação e executar.

**Obs**: Não posso ainda fornecer os dados respectivos aos sons da ave [S. Albilora](http://www.wikiaves.com.br/wiki/joao-do-pantanal), nem dos *backgrounds*, assim que eu tiver a permissão retiro essa observação e atualizo o [settings.sh](/settings.sh).

**Obs2**: A versão digitalizada do TCC ainda não está disponível, até final de Julho será colocada aqui.

O [settings.sh](/settings.sh) cria as pastas com os requerimentos de dados e modelo necessário neste trabalho, com uma estrutura da seguinte forma:

```
data/
    S-Albilora
    features
    ...
models/
    CNN14.pth
```

Em que features é o padrão para salvar os dados os quais serão utilizados para o treinamento, gerados a partir do [pack_audio_files_to_hdf5](/panns/utils/pack_audio_files_to_hdf5.py).

Tanto pelo anaconda com python 3.10, quanto somente com o python 3.10, eu não encontrei problemas até o momento com os códigos. Talvez hajam algumas incompatibilidades que não cheguei a encontrar, se você encontrar, informe com uma [issue](https://github.com/SousaPedroso/PIDL/issues/new).