# A Empty Model

Build a model of tf1 quickly from scratch. The project contains the tconfig reader, data preprocess and model runner.

## Requirements
- python>=3.6
- tensorflow>=1.12.0
- numpy


Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Model Training
```bash
python train.py --config_filename={config_filename}
```


## Model Evaluating
```bash
python eval.py --config_filename={saved_model_config_filename}
```