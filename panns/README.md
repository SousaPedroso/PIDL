Este módulo contém todos os processos necessários para reproduzir experimentos
com as [PANNs](https://ieeexplore.ieee.org/abstract/document/9229505/). Aqui listarei os principais comandos e as informações necessárias antes de executar alguns arquivos.
Para qualquer atividade aqui dentro, primeiro instale, configure e ative o ambiente necessário, a flag ```--dev``` para o primeiro comando é opcional se quiser instalar o pylint:

```bash
pipenv install .
pipenv shell
```

Iniciar o servidor (local) responsável por rastrear os experimentos, o que inclui configurações dos áudios (com exceção de duração), bem como as métricas, modelo treinado e suas dependências. Também usei durante o TCC o [azure machine learning](https://azure.microsoft.com/pt-br/free/machine-learning/search/?ef_id=_k_CjwKCAjwvJyjBhApEiwAWz2nLVrpda3z9uErAMWV4d7Q59u9F6o1XesXLifAyWIZT8tbgMQ3KwjqgBoCTk8QAvD_BwE_k_&OCID=AIDcmmzmnb0182_SEM__k_CjwKCAjwvJyjBhApEiwAWz2nLVrpda3z9uErAMWV4d7Q59u9F6o1XesXLifAyWIZT8tbgMQ3KwjqgBoCTk8QAvD_BwE_k_&gclid=CjwKCAjwvJyjBhApEiwAWz2nLVrpda3z9uErAMWV4d7Q59u9F6o1XesXLifAyWIZT8tbgMQ3KwjqgBoCTk8QAvD_BwE) para executar em paralelo alguns experimentos. Nesse caso, não é necessário esse comando abaixo. Usei somente uma VM ubuntu com 2 núcleos, sem GPU e com 28GB de RAM. Essa quantidade foi para poder executar os experimentos com todas as entradas do conjunto de validação, no entanto, com os créditos chegando ao fim, optei por mudar alguns métodos para realizar processamento em lote. Se você é estudante e quiser, veja se está [apto](https://education.github.com/) a receber U$100 créditos da microsoft azure para executar os experimentos. 

```bash
mlflow server --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "file:///abspath/PIDL/panns/mlruns"
```

Para treinar modelos, o código abaixo pode ser executado. Fique atento quanto a alguns parâmetros adicionais que queira passar, principalmente algumas combinações de parâmetros para os espectrogramas que não são possíveis devido a entrada esperada pelas _PANNs_:

```bash
python train_model.py --model_iterations 100 --train_path "worskpace_dir\features\waveform.h5" --experiment_name "PANNs default settings" --pretrained_path "pretrained\models\Cnn14.pth"
```

Meus experimentos com os métodos de interpretabilidades estão separados em 3 notebooks,
da mesma forma que separei no TCC: 1 para métodos baseados em gradientes locais, outro para métodos baseados em referência/conceitos e por último o método baseado em pertubação _LIME_. Todos na pasta de [interpretabilidade](/panns/interpretability/). Além disso, há alguns **scripts** e arquivos úteis para coordenar os notebooks e ideias obtidas através da interpretabilidade.

Os métodos baseados em gradiente foram usados para ter uma visão geral do comportamento do modelo em cada experimento, enquanto os métodos baseados em referência foram utilizados para maior entendimento da influência das entradas e diferentes características dos áudios para o modelo. Por último, o [LIME](https://arxiv.org/abs/1602.04938) foi pensado em específico para trabalhar com o S. albilora e entender algumas variações do que foi aprendido pelo modelo.