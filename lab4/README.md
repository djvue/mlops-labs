# HW4: dvc with titanic dataset

## Описание шагов

- `load_dataset.py` скачивает датасет titanic из `catboost.datasets`  и сохраняет в директорию `data`
- `modify_dataset_1.py` минимально модифицирует датасет (удаляет строки 1, 2, 3)
- `modify_dataset_2.py` заменяет отсутствующие значения колонки Age на среднее значение
- `modify_dataset_3.py` заменяет колонку `Sex` на one-hot-encoding колонки `Sex_female` и `Sex_male`

## DVC

Результат каждого шага зафиксирован в DVC и git.

DVC привязан к google:
https://drive.google.com/drive/folders/1MYqLkrON3SsLHTyNkP_shRiLZ1qsgTAV?usp=sharing

### Запуск скриптов

Установка зависимостей
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Модификация датасета
```sh
python load_dataset.py
python modify_dataset_1.py
python modify_dataset_2.py
python modify_dataset_3.py
```
