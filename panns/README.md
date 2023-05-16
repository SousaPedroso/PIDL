Este módulo contém todos os processos necessários para reproduzir experimentos
com as [PANNs](https://ieeexplore.ieee.org/abstract/document/9229505/).
Para qualquer atividade aqui dentro, primeiro instale e configure o ambiente necessário e o ative:

```bash
pipenv install .
pipenv shell
```

Iniciar o servidor (local) responsável por rastrear os experimentos, o que inclui configurações dos áudios (com exceção de duração), bem como as métricas, modelo treinado e suas dependências.
```bash
mlflow server --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "file:///abspath/PIDL/panns/mlruns"
```

Para treinar modelos:
```bash
python train_model.py --model_iterations 100 --train_path "worskpace_dir\features\waveform.h5" --experiment_name "PANNs default settings" --pretrained_path "pretrained\models\Cnn14.pth"
```

Para gerar métricas de interpretabilidade:
```bash
cd interpretability
python report_metrics.py --dataset_dir "data\S-Albilora" --workspace "workspace\features\waveform.h5" --run_id run_id
```

Meus experimentos com os métodos de interpretabilidades estão separados em 3 notebooks,
da mesma forma que separei em TCC: 1 para métodos baseados em gradientes locais, outro para métodos baseados em referência/conceitos e por último o método baseado em pertubação LIME. Todos na pasta de [interpretabilidade](/panns/interpretability/).

Os métodos baseados em gradiente foram usados para ter uma visão geral do comportamento do modelo em cada experimento, enquanto os métodos baseados em referência foram utilizados para maior entendimento da influência das entradas e diferentes características dos áudios para o modelo. Por último, o [LIME](/panns/interpretability/pertubation.ipynb) foi pensado em específico para trabalhar com o S. albilora e entender algumas variações do que foi aprendido pelo modelo.