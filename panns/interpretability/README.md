Para gerar métricas de interpretabilidade, o código abaixo pode ser realizado. Aqui, eu digo métricas, pois é somente a visão geral do modelo para qualquer uma das subdivisões da base de dados (treinamento e validação) em **folders**, com algumas métricas a mais além das que foram utilizadas como rastreio durante o treinamento do modelo:
```bash
cd interpretability
python report_metrics.py --dataset_dir "data\S-Albilora" --workspace "workspace\features\waveform.h5" --run_id run_id
```

O script [move_audios_by_attribution.py](/panns/interpretability/move_audios_by_attribution.py), é com base no que escrevi para remover entradas cuja influência prejudique o modelo. O [clusteryze_species.py](/panns/interpretability/clusteryze_species.py) é somente para tentar separar diferentes espécies de aves, com rápida velocidade de processamento, somente com o PCA do espectrograma considerando duas dimensões. Ou seja, ao invés de remover as entradas, só dividimos em outras sub_classes.