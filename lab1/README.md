# MLOps. Практическое задание №1

## Getting started

### Установка рабочего окружения

Для работы необходим python и venv. Проверена работа на python 3.11

```sh
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

### Запуск пайплайна

```sh
./pipeline.sh
```

Если недостаточно прав на запуск скрипта (возможно отключен git core.filemode), нужно добавить права `chmod +x ./pipeline.sh`

### Запуск в docker-е

```sh
docker build -t mlops_hw1 .
docker run --rm mlops_hw1

# Или одной командой
docker run --rm $(docker build -q .)
```

### Ожидаемый вывод
```
data_creation.py: data created successfully
model_preprocessing.py: data preprocessed successfully
model_preparation.py: model trained successfully
model metrics:
{'mse': 1.0012895858737432, 'r2': 0.999867902012933}
model_testing.py: model tested successfully
```

## Описание пайплайна

Пайплайн состоит из 4 стадий:
- data_creation.py: генерация данных, разделение на тренировочную и тестовую выбори и сохранение в `data/train/raw.csv` и `data/test/raw.csv`
- model_preprocessing.py: предобработка данных с использованием `StandardScaler`. Данные сохраняются в `data/train/preprocessed.csv` и `data/test/preprocessed.csv`
- model_preparation.py: обучение модели линейной регрессии на тренировочных данных. Для сохранения модели используется joblib, файл `data/model.joblib`
- model_testing.py: тестирование модели и подсчет метрик - mse и r2_score. Метрики выводятся в консоль

Тестирование модели дает адекватные результаты, точность близка к 100%.
