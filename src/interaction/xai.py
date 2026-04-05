from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import build_feature_frame_for_split
from .train_model import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, load_interaction_model


def _prepare_split_matrix(
    model: Any,
    data_dir: str | Path,
    split: str,
    sample_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_df = build_feature_frame_for_split(model=model, data_dir=data_dir, split=split)
    if sample_size > 0 and len(feature_df) > sample_size:
        feature_df = feature_df.sample(sample_size, random_state=42).reset_index(drop=True)

    texts = (feature_df["text_input"] + " [URL] " + feature_df["url_input"]).tolist()
    embeddings = model._extract_embeddings(texts)
    x_struct = model.preprocessor.transform(feature_df[model.structured_cols])
    x_all = np.hstack([x_struct, embeddings]).astype(np.float32)
    return feature_df, x_all


def _select_malicious_shap(shap_values: Any, model: Any) -> np.ndarray:
    mal_idx = model.label_to_id["malicious"]
    if isinstance(shap_values, list):
        return np.array(shap_values[mal_idx])

    arr = np.array(shap_values)
    if arr.ndim == 3:
        if arr.shape[2] == len(model.label_to_id):
            return arr[:, :, mal_idx]
        if arr.shape[0] == len(model.label_to_id):
            return arr[mal_idx]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


def _select_expected_value(explainer: Any, model: Any) -> float:
    mal_idx = model.label_to_id["malicious"]
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev_arr = np.array(ev).reshape(-1)
        if len(ev_arr) > mal_idx:
            return float(ev_arr[mal_idx])
    return float(ev) if not isinstance(ev, list) else float(ev[0])


def compute_shap_values(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    data_dir: str | Path = "interaction_data",
    split: str = "test_ood",
    sample_size: int = 200,
) -> dict[str, Any]:
    model = load_interaction_model(model_dir=model_dir)
    feature_df, x_all = _prepare_split_matrix(model=model, data_dir=data_dir, split=split, sample_size=sample_size)

    explainer = shap.TreeExplainer(model.xgb_model, approximate=True)
    shap_values = explainer.shap_values(x_all)
    malicious_shap = _select_malicious_shap(shap_values=shap_values, model=model)
    expected_value = _select_expected_value(explainer=explainer, model=model)

    return {
        "model": model,
        "feature_df": feature_df,
        "x_all": x_all,
        "feature_names": model.full_feature_names,
        "explainer": explainer,
        "malicious_shap": malicious_shap,
        "expected_value": expected_value,
    }


def _fallback_summary_plot(feature_names: list[str], values: np.ndarray, path: Path) -> None:
    top_n = min(20, len(feature_names))
    idx = np.argsort(np.abs(values))[::-1][:top_n]
    names = [feature_names[i] for i in idx][::-1]
    scores = values[idx][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(names, scores)
    plt.xlabel("Contribution / Importance")
    plt.title("Interaction Feature Contribution Summary")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _fallback_force_html(feature_names: list[str], values: np.ndarray, path: Path) -> None:
    top_n = min(20, len(feature_names))
    idx = np.argsort(np.abs(values))[::-1][:top_n]
    rows = "\n".join(
        f"<tr><td>{feature_names[i]}</td><td>{values[i]:.6f}</td></tr>"
        for i in idx
    )
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Interaction XAI Force Plot</title></head>
<body>
<h2>Interaction Model Contribution Table (Fallback)</h2>
<p>SHAP interactive force plot could not be generated; this fallback lists top contributions.</p>
<table border="1" cellpadding="6" cellspacing="0">
<thead><tr><th>Feature</th><th>Contribution</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def export_xai_artifacts(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    data_dir: str | Path = "interaction_data",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "test_ood",
    sample_size: int = 200,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "shap_summary.png"
    force_path = out_dir / "shap_force_plot.html"

    result: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "force_plot_path": str(force_path.resolve()),
        "backend": "shap",
    }

    try:
        payload = compute_shap_values(
            model_dir=model_dir,
            data_dir=data_dir,
            split=split,
            sample_size=sample_size,
        )
        malicious_shap = payload["malicious_shap"]
        x_all = payload["x_all"]
        feature_names = payload["feature_names"]
        explainer = payload["explainer"]
        expected_value = payload["expected_value"]

        shap.summary_plot(
            malicious_shap,
            x_all,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=180)
        plt.close()

        force_plot = shap.force_plot(
            expected_value,
            malicious_shap[0],
            x_all[0],
            feature_names=feature_names,
            matplotlib=False,
        )
        shap.save_html(str(force_path), force_plot)
        result["backend"] = "shap_tree_explainer"
        result["sample_size"] = int(x_all.shape[0])
    except Exception as exc:
        model = load_interaction_model(model_dir=model_dir)
        feature_importances = np.array(model.xgb_model.feature_importances_)
        if feature_importances.size == 0:
            feature_importances = np.zeros(len(model.full_feature_names), dtype=float)

        _fallback_summary_plot(
            feature_names=model.full_feature_names,
            values=feature_importances,
            path=summary_path,
        )
        _fallback_force_html(
            feature_names=model.full_feature_names,
            values=feature_importances,
            path=force_path,
        )
        result["backend"] = "fallback"
        result["warning"] = str(exc)

    manifest_path = out_dir / "xai_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest_path"] = str(manifest_path.resolve())
    return result
