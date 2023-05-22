# _Post-hoc_ Interpretable Deep Learning for Bioacustics data (PIDL)

Esse repositório contém os códigos para os experimentos utilizados para a construção do meu trabalho de conclusão de curso (TCC) em [Ciência da Computação](https://www.ic.ufmt.br/) pela [UFMT](https://www.ufmt.br/). A seguir mostro a tabela de seções do repositório e depois o que contém cada uma.

## Tabela de seções

- [Motivação](#motivação)
- [Estrutura](#estrutura)
- [Configuração](#configuração)

## Motivação

Ao longo de minha iniciação científica (IC) na graduação, encontrei alguns _papers_ a respeito de  **arquiteturas interpretáveis** de _DL_ (e.g. [Prototype Learning](https://ieeexplore.ieee.org/document/9747014/)) para sons bem como métodos de interpretabilidade para Redes Neurais com modelos já treinados, e.g [DeepLift](https://arxiv.org/abs/1704.02685). Para ajudar mais o grupo de pesquisa [CO.BRA](https://cobra.ic.ufmt.br/) ao qual participei com IC e entender também algumas dificuldades ao longo destes anos, decidi seguir com esse tópico de estudo, por ser algo ainda não trabalhado. Há muito espaço para melhorias com interpretabilidade não só na bioacústica, como em outras áreas, mas por ser um trabalho de conclusão de curso, me limitei a algo pouco explorado na bioacústica (métodos _post-hoc_), mas que não fosse tão simples de entender/aplicar. Dentre algumas outras opções, podem ser realizados testes com outras bases de dados bem como outros modelos para maior certificação da importância de interpretabilidade ou **possível** generalização de características aprendidas. Explico com mais detalhes em meu TCC.

## Estrutura

Esta seção é para detalhar como foi pensada a estrutura de diretórios. O diretório _PANNs_ consiste de um ambiente cujas dependências são controladas através do [pipenv](https://github.com/pypa/pipenv). Mudanças nos parâmetros do modelo são rastreadas através do [mlflow](https://github.com/mlflow/mlflow). Mudanças em processamentos podem ser feitas com anotações através de _tags_ (ou ainda salvando arquivos necessários, que não foi preciso aqui). Dentro de [panns](/panns/), a pasta [utils](/panns/utils/) contém códigos auxiliares necessários para todos os módulos.

## Configuração

Esta seção mostra como você pode reproduzir meus experimentos. Não somente o treinamento, validação e teste dos modelos são reproduzíveis, mas a ideia é também permitir um ambiente reproduzível para que você precise somente fazer a instalação e rodar!