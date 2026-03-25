import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

def _numeric_preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)
    return X_scaled_df

def _obj_preprocessing(X):
    le = LabelEncoder()
    X_obj = X.select_dtypes(include=[object]).copy()
    for col in X_obj.columns:
        X_obj[col] = le.fit_transform(X_obj[col].astype(str))
    # На инференсе колонка `nm` может отсутствовать (или быть вынесена в другой тип).
    # Чтобы пайплайн не падал из-за схемы входных данных, удаляем её "мягко".
    X_obj = X_obj.drop(["nm"], axis=1, errors="ignore")
    return X_obj

def _target_preprocessing(y):
    y_classes = np.where(y < 2.5, 0, 1)
    return y_classes

def _remove_blowouts(X, y):
    criteria = y.std() * 3
    y_filtered = y[y > criteria]
    X_filtered = X[y > criteria]
    return X_filtered, y_filtered

def preprocessing(X, *, dropna: bool = False, drop_duplicates: bool = False):
    # Для инференса обычно критично не менять число строк (иначе предикты
    # не сопоставляются обратно с исходными данными). Поэтому "dropna" и
    # "drop_duplicates" делаем опциональными.
    data = X.copy()

    if dropna:
        data = data.dropna(axis=0)
    else:
        # StandardScaler не умеет NaN, поэтому явно валидируем только числовые колонки.
        num_nan = data.select_dtypes(include=[np.number]).isna().any().any()
        if num_nan:
            raise ValueError(
                "В числовых признаках обнаружены NaN. Для инференса включите dropna=True "
                "или заранее обработайте пропуски во входных данных."
            )

    if drop_duplicates:
        data = data.drop_duplicates()
    
    # Числовые признаки
    num_cols = data.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        scaled_num = _numeric_preprocess(data[num_cols])
    else:
        scaled_num = pd.DataFrame(index=data.index)

    # Категориальные (объектные) признаки
    obj_cols = data.select_dtypes(include=[object]).columns
    if len(obj_cols) > 0:
        encoded_obj = _obj_preprocessing(data[obj_cols])
    else:
        encoded_obj = pd.DataFrame(index=data.index)

    # Объединить обратно
    result = pd.concat([scaled_num, encoded_obj], axis=1)
    return result
