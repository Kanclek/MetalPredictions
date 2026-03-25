import pandas as pd
import numpy as np


def _get_params(PATH):
    params = pd.read_excel(PATH)

    # Формируем "№ патрубка": 4 цифры из "ID", 1 из "nr", 2 из "nz" (с лидирующим нулём при необходимости)
    params["№ патрубка"] = (
        params["ID"].astype(str).str[-4:] +  # последние 4 символа
        params["nr"].astype(str) +           # номер ручья
        params["nz"].astype(str).str.zfill(2) # номер заготовки (2 цифры, с лид. нулём)
    )
    return params

def _get_target(PATH):
    targets = []
    sheet_names = pd.ExcelFile(PATH).sheet_names
    for name in sheet_names:
        target = pd.read_excel(PATH, sheet_name=name)
        if not "Силикаты недеформированные (2009)" in target.columns:
            continue
        targets.append(target[["Марка стали", "Диаметр", "№ патрубка", "Силикаты недеформированные (2009)"]])
    target_result = pd.concat(targets, axis=0)

    target_result_grouped = target_result.groupby("№ патрубка", as_index=False).agg({
    "Марка стали": "first",
    "Диаметр": "first",
    "Силикаты недеформированные (2009)": "max"
    })
    target_result_grouped = target_result_grouped.dropna(axis=0)
    return target_result_grouped


def form_data(params_path, target_path):
    params = _get_params(params_path)
    target = _get_target(target_path)
    return pd.merge(params, target, on="№ патрубка", how="inner").drop(["ID", "№ патрубка"], axis=1)

