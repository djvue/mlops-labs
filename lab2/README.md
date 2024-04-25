# MLOps. Практическое задание №2

## Описание содержания этапов пайплайна

Используется dataset https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv.
Он представляет собой параметры домов в различных расположениях и целевую переменную - цену продажи на дом.

Для препроцессинга используются StandardScaler, PowerTransformer для числовых признаков
и OrdinalEncoder, RareGrouper, OneHotEncoder для категориальных признаков, подобранные при анализе данных.

В качестве модели выбрана XGBoost, так как дает лучшие результаты.

Результаты тестирования модели:
```
{'mse': 661261203.8071303, 'r2': 0.8929128628634568}
```

### Вывод:
Модель дает приемлемые результаты, учитывая большой шум данных, малое количество данных.

Для дальнейшего улучшения результатов можно рассмотреть другие модели, например нейросети,
и более качественный подбор гиперпараметров моделей, например с использованием RandomizedSearchCV или GridSearchCV.
Также имеет смысл для анализа использовать кросс-валидацию для более точной оценки.

## Описание пайплайна и среды выполнения

Код пайплайна лежит в файле Jenkinsfile. Используются плагины jenkins git, docker, docker-pipeline, kubernetes.

Стенд, на котором тестировался пайплайн развернут в среде kubernetes.
К master-ноде jenkins подключено облако ("Clouds") kubernetes в том же кластере.
В пайплайне Jenkinsfile указан образ docker ("podTemplate", "containerTemplate", "container"),
на котором запускатся шаги пайплайна.

- podTemplate, containerTemplate, container: настройка окружения для сборки. 
В данном случае используется docker-образ `python:slim`.
- git url: checkout репозитория с кодом `https://github.com/djvue/mlops-labs` (текущий репозиторий)
- Install python packages: установка пакетов-зависимостей python из requirements.txt
- Download dataset - `data_loading.py` - скачивание датасета, залитого на github
- Preprocess data - `data_preprocessing.py` - предобработка данных, разделение на тренировочный и тестовый датасет
- Model training - `model_training.py` - обучение модели XGBoost на тренировочных данных
- Model testing - `model_testing.py` - тестирование модели на тестовых данных
