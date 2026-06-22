"""
evaluation.py — Métricas de validação e visualização de resultados.
Etapa 8 do pipeline LULC.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix,
)


def compute_metrics(y_true, y_pred, class_names):
    """
    Calcula acurácia global, Kappa de Cohen e F1-score por classe.

    Retorna:
        metrics:   dict com accuracy, kappa, f1_macro, f1_weighted
        df_report: DataFrame com precision/recall/f1 por classe (sklearn classification_report)
    """
    metrics = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'kappa':       cohen_kappa_score(y_true, y_pred),
        'f1_macro':    f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    df_report = pd.DataFrame(report).T
    return metrics, df_report


def plot_confusion_matrix(y_true, y_pred, class_names, title='Matriz de Confusão', normalize=True):
    """
    Plota matriz de confusão com seaborn heatmap.
    normalize=True mostra recall por classe (proporção por linha).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = '.2f', 1.0
    else:
        cm_plot, fmt, vmax = cm, 'd', cm.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=vmax, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real (MapBiomas)')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, title='Importância das Features'):
    """
    Plota importância das features (sklearn RF) ordenada de forma decrescente.
    """
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in idx],
        [importances[i] for i in idx],
        color='steelblue',
    )
    ax.set_xlabel('Importância (mean decrease impurity)')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def compare_models(results: dict, metric='f1_macro'):
    """
    Plota comparação de uma métrica entre múltiplos modelos.
    results: {'NomeModelo': {'accuracy': ..., 'kappa': ..., 'f1_macro': ...}, ...}
    """
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(names, values, color=['#2d6fd3', '#d4a028', '#1f7a1f'][:len(names)])
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparação de modelos — {metric}')
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    return fig
