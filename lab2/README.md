# MLOps. Практическое задание №2

## Описание содержания этапов пайплайна

Используется dataset https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv.
Он представляет собой параметры домов в различных расположениях и целевую переменную - цену продажи на дом.

Для препроцессинга используются StandardScaler, PowerTransformer для числовых признаков
и OrdinalEncoder, RareGrouper, OneHotEncoder для категориальных признаков, подобранные при анализе данных.

В качестве модели выбрана XGBoost, так как дает лучшие результаты.

Для запуска необходимо скачать файлы в папку data рядом с ipynb файлом (хотя бы `data/train.csv`)

### Описание пайплайна и среды выполнения

Код пайплайна лежит в файле Jenkinsfile. Используются плагины jenkins git, docker, docker-pipeline, kubernetes.

Стенд, на котором тестировался пайплайн развернут в среде kubernetes.
В пайплайне указан образ docker, на котором запускатся шаги пайплайна.

- Install python packages: установка пакетов-зависимостей python из requirements.txt
- Download dataset - data_loading.py - скачивание датасета, залитого на github
- Preprocess data - data_preprocessing.py - предобработка данных, разделение на тренировочный и тестовый датасет
- Model training - model_training.py - обучение модели XGBoost на тренировочных данных
- Model testing - model_testing.py - тестирование модели на тестовых данных
