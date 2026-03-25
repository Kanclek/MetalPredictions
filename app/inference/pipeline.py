from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from .model.nn import SimpleNeuralNetwork
from .preprocessing.data_preprocessing import preprocessing as preprocess_features
from .preprocessing.extraction import form_data

LABEL_COL = "Силикаты недеформированные (2009)"


@dataclass(frozen=True)
class InferenceConfig:
    weights_path: str
    device: Optional[str] = None  # "cpu" / "cuda" / None -> auto
    nz_value: Optional[Union[int, float]] = None
    batch_size: int = 512


class NNInferencePipeline:
    """
    NN пайплайн для инференса:
    1) склеиваем входные Excel (params+target) в DataFrame
    2) при необходимости фильтруем по `nz`
    3) приводим признаки к формату модели (scaler+label encoding из текущего пайплайна)
    4) выполняем forward и возвращаем предикты/вероятности
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Загружаем state_dict и автоматически восстанавливаем архитектуру из весов.
        state = torch.load(config.weights_path, map_location="cpu")
        self.model, self.input_size, self.num_classes = self._build_model_from_state(state)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_model_from_state(state_dict: dict) -> tuple[torch.nn.Module, int, int]:
        if "fc_out.bias" not in state_dict:
            raise ValueError("Не найден ключ `fc_out.bias` в state_dict. Неподдерживаемый формат весов.")

        num_classes = int(state_dict["fc_out.bias"].shape[0])

        # Линейные слои: у Linear weight тензор 2D, у BatchNorm weight обычно 1D.
        linear_weights: list[tuple[int, torch.Tensor]] = []
        for k, v in state_dict.items():
            if not (k.startswith("hidden_layers.") and k.endswith(".weight")):
                continue
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            # k вида "hidden_layers.{idx}.weight"
            idx = int(k.split(".")[1])
            linear_weights.append((idx, v))

        if not linear_weights:
            raise ValueError("Не удалось восстановить размеры hidden_layers из state_dict.")

        linear_weights.sort(key=lambda x: x[0])
        # weight Linear: [out_features, in_features]
        input_size = int(linear_weights[0][1].shape[1])
        hidden_sizes = [int(w.shape[0]) for _, w in linear_weights]

        model = SimpleNeuralNetwork(
            input_size=input_size,
            num_classes=num_classes,
            # dropout_rate влияет только на поведение в train, в eval оно выключено.
            dropout_rate=0.3,
            hidden_sizes=hidden_sizes,
        )
        model.load_state_dict(state_dict, strict=True)
        return model, input_size, num_classes

    def _prepare_features(
        self, merged_df: pd.DataFrame, nz_value: Optional[Union[int, float]]
    ) -> torch.Tensor:
        df = merged_df.copy()

        if nz_value is not None:
            if "nz" not in df.columns:
                raise ValueError("Колонка `nz` отсутствует во входных данных, но задан `nz_value`.")
            df = df[df["nz"] == nz_value].reset_index(drop=True)

        X = df
        if LABEL_COL in X.columns:
            X = X.drop(columns=[LABEL_COL])
        if "nz" in X.columns:
            X = X.drop(columns=["nz"])

        X_pre = preprocess_features(X)
        if X_pre.shape[1] != self.input_size:
            raise ValueError(
                f"Несовпадение размерности признаков: model ждёт input_size={self.input_size}, "
                f"а preprocessing вернул shape={tuple(X_pre.shape)}."
            )

        # Модель ждёт float32 тензор.
        return torch.tensor(X_pre.values, dtype=torch.float32)

    @torch.no_grad()
    def predict_from_merged_df(
        self, merged_df: pd.DataFrame, nz_value: Optional[Union[int, float]] = None
    ) -> pd.DataFrame:
        if nz_value is None:
            nz_value = self.config.nz_value

        features = self._prepare_features(merged_df, nz_value=nz_value).to(self.device)

        preds_list: list[np.ndarray] = []
        probs_list: list[np.ndarray] = []

        for start in range(0, features.shape[0], self.config.batch_size):
            batch = features[start : start + self.config.batch_size]
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_list.append(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

        preds_np = np.concatenate(preds_list, axis=0)
        probs_np = np.concatenate(probs_list, axis=0)

        out = merged_df.copy()
        if nz_value is not None:
            out = out[out["nz"] == nz_value].reset_index(drop=True)

        # В модели `nz` не используется, поэтому удаляем колонку из результата,
        # чтобы соответствовать логике обучения.
        if "nz" in out.columns:
            out = out.drop(columns=["nz"])

        out["pred_class"] = preds_np
        if self.num_classes == 2:
            out["pred_proba_class_1"] = probs_np[:, 1]
        else:
            # Для многокласса сохраняем max probability.
            out["pred_proba_max"] = probs_np.max(axis=1)

        return out

    def predict(
        self,
        params_path: str,
        target_path: str,
        nz_value: Optional[Union[int, float]] = None,
    ) -> pd.DataFrame:
        merged_df = form_data(params_path=params_path, target_path=target_path)
        return self.predict_from_merged_df(merged_df, nz_value=nz_value)


def build_pipeline(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    nz_value: Optional[Union[int, float]] = None,
    batch_size: int = 512,
) -> NNInferencePipeline:
    if weights_path is None:
        app_root = Path(__file__).resolve().parents[1]  # .../app/inference/pipeline.py -> .../app
        candidate = app_root / "inference" / "model" / "repository" / "model2_weights.pth"
        if not candidate.exists():
            raise FileNotFoundError(f"Не найден файл весов по пути: {candidate}")
        weights_path = str(candidate)

    return NNInferencePipeline(
        InferenceConfig(
            weights_path=weights_path,
            device=device,
            nz_value=nz_value,
            batch_size=batch_size,
        )
    )

