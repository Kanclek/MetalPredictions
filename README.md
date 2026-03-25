## metal_prediction / project

### Что здесь лежит

- **`example.ipynb`**: минимальный пример, как вызвать `build_pipeline()` и получить предикт.
- **`app/`**: копия пакета `app` (инференс-логика + препроцессинг), чтобы пример работал из `project/` без лишних зависимостей от внешней структуры.

### Быстрый старт (Jupyter)

Открой `project/example.ipynb`, подставь пути к данным и запусти:

- **`params_path`**: Excel с параметрами плавки/разливки (как в `extraction._get_params`)
- **`target_path`**: Excel с таргетом/листами (как в `extraction._get_target`)

В ноутбуке используется:

- `from app.inference.pipeline import build_pipeline`
- `tmp = build_pipeline()`
- `df_pred = tmp.predict(params_path=..., target_path=...)`

### Запуск через CLI (если используешь `app/main.py`)

Если запускаешь из корня репозитория:

```bash
python app/main.py --params "D:/.../mnls_od_2025.xlsx" --target "D:/.../a_ha_2025.xlsx"
```

Опционально можно сохранить результат:

```bash
python app/main.py --params "D:/.../mnls_od_2025.xlsx" --target "D:/.../a_ha_2025.xlsx" --out "D:/.../pred.csv"
```

### Где берутся веса

По умолчанию веса **не нужно** передавать вручную: `build_pipeline()` автоматически ищет файл:

- `app/inference/model/repository/model2_weights.pth`

Если файла нет, будет понятная ошибка `FileNotFoundError` с ожидаемым путём.

### Что возвращает инференс

`predict()` возвращает `pandas.DataFrame` со всеми исходными колонками + предсказания:

- **`pred_class`**: предсказанный класс (0/1)
- **`pred_proba_class_1`**: вероятность класса 1 (для бинарной классификации)

Примечание: колонка `nz` **не используется** моделью как признак и на выходе удаляется (по логике обучения в `source`).

