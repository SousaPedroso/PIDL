Este módulo contém todos os processos necessários para reproduzir experimentos
com as [PANNs](https://ieeexplore.ieee.org/abstract/document/9229505/).
Para qualquer atividade aqui dentro, primeiro instale e configure o ambiente necessário e o ative:

```bash
pipenv install .
pipenv shell
```

Iniciar o servidor (local) responsável por rastrear os experimentos, o que inclui configurações dos áudios (com exceção de duração), bem como as métricas, modelo treinado e suas dependências
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