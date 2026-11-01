"""mag_nav/ml/metrics.py

Метрики позиционирования и ориентации для навигации по магнитному маяку.

Публичный API
─────────────
evaluate_predictions(...)       – сводная таблица 3-D ошибки (совместимо со старой версией)
distance_bin_metrics(...)       – Mean/Median/P95 по бинам расстояния (совместимо)
component_metrics(...)          – ошибка по осям x, y, z отдельно
orientation_metrics(...)        – угловая ошибка дипольного момента (если есть)
training_diagnostics(...)       – сходимость, переобучение, best-epoch из history
improvement_stats(...)          – % улучшения, Wilcoxon, доля «выбросов»
print_report(run_dir)           – ЕДИНЫЙ консольный отчёт по каталогу запуска

Все функции чистые (возвращают DataFrame / dict), кроме print_report,
который только печатает и возвращает None.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# ──────────────────────────────────────────────
# Внутренние утилиты
# ──────────────────────────────────────────────

_LINE = "─" * 78
_DLINE = "═" * 78


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _summary(error: ArrayLike) -> dict[str, float]:
    """Базовые статистики одномерного вектора ошибок."""
    e = _finite(np.asarray(error, dtype=np.float64).ravel())
    if e.size == 0:
        return dict(
            n=0,
            mean_m=np.nan,
            median_m=np.nan,
            rmse_m=np.nan,
            std_m=np.nan,
            p90_m=np.nan,
            p95_m=np.nan,
            p99_m=np.nan,
            max_m=np.nan,
        )
    return dict(
        n=int(e.size),
        mean_m=float(np.mean(e)),
        median_m=float(np.median(e)),
        rmse_m=float(np.sqrt(np.mean(e**2))),
        std_m=float(np.std(e, ddof=1)) if e.size > 1 else 0.0,
        p90_m=float(np.percentile(e, 90)),
        p95_m=float(np.percentile(e, 95)),
        p99_m=float(np.percentile(e, 99)),
        max_m=float(np.max(e)),
    )


def _fmt(v: Any, unit: str = "m", decimals: int = 4) -> str:
    """Человекочитаемое число с единицей."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "   –"
    return f"{v:.{decimals}f} {unit}"


def _table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], *, align: str = ">", decimals: int = 4
) -> str:
    """Простая ASCII-таблица без внешних зависимостей."""
    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append(
            [
                _fmt(c, decimals=decimals)
                if isinstance(c, (int, float, np.floating, np.integer))
                else str(c)
                for c in row
            ]
        )
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line_row(sep: str = "─", cross: str = "┼") -> str:
        return cross.join(sep * (w + 2) for w in widths)

    def _data_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if align == ">" else cell.ljust(widths[i]))
        return "│ " + " │ ".join(parts) + " │"

    out = [_line_row(), _data_row(headers), _line_row()]
    for row in str_rows:
        out.append(_data_row(row))
    out.append(_line_row())
    return "\n".join(out)


def _section(title: str) -> str:
    return f"\n{_DLINE}\n  {title}\n{_DLINE}"


# ──────────────────────────────────────────────
# 1. Позиционирование: 3-D ошибка
# ──────────────────────────────────────────────


def evaluate_predictions(
    true_positions: ArrayLike,
    dipole_positions: ArrayLike,
    corrected_positions: ArrayLike,
) -> pd.DataFrame:
    """Сводные метрики базовой и скорректированной 3-D ошибки (м).

    Совместимо по сигнатуре с прежней версией модуля.
    """
    true = np.asarray(true_positions, dtype=np.float64).reshape(-1, 3)
    base = np.asarray(dipole_positions, dtype=np.float64).reshape(-1, 3)
    corr = np.asarray(corrected_positions, dtype=np.float64).reshape(-1, 3)
    if true.shape != base.shape or true.shape != corr.shape:
        raise ValueError("true, baseline и corrected должны иметь форму (n, 3)")
    keep = (
        np.all(np.isfinite(true), 1) & np.all(np.isfinite(base), 1) & np.all(np.isfinite(corr), 1)
    )
    e_b = np.linalg.norm(true[keep] - base[keep], axis=1)
    e_c = np.linalg.norm(true[keep] - corr[keep], axis=1)
    rows = [
        {"estimator": "Baseline (Dipole)", **_summary(e_b)},
        {"estimator": "Corrected (Dipole+ML)", **_summary(e_c)},
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 2. Ошибки по компонентам x, y, z
# ──────────────────────────────────────────────


def component_metrics(
    true_positions: ArrayLike,
    baseline_positions: ArrayLike,
    corrected_positions: ArrayLike,
) -> pd.DataFrame:
    """MAE / RMSE / P95 по каждой координатной оси."""
    true = np.asarray(true_positions, dtype=np.float64).reshape(-1, 3)
    base = np.asarray(baseline_positions, dtype=np.float64).reshape(-1, 3)
    corr = np.asarray(corrected_positions, dtype=np.float64).reshape(-1, 3)
    keep = (
        np.all(np.isfinite(true), 1) & np.all(np.isfinite(base), 1) & np.all(np.isfinite(corr), 1)
    )
    true, base, corr = true[keep], base[keep], corr[keep]
    rows: list[dict] = []
    for i, axis in enumerate("xyz"):
        db = true[:, i] - base[:, i]
        dc = true[:, i] - corr[:, i]
        rows.append(
            {
                "axis": axis,
                "baseline_mae": float(np.mean(np.abs(db))),
                "baseline_rmse": float(np.sqrt(np.mean(db**2))),
                "baseline_p95": float(np.percentile(np.abs(db), 95)),
                "corrected_mae": float(np.mean(np.abs(dc))),
                "corrected_rmse": float(np.sqrt(np.mean(dc**2))),
                "corrected_p95": float(np.percentile(np.abs(dc), 95)),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. Биннинг по расстоянию
# ──────────────────────────────────────────────


def distance_bin_metrics(
    true_positions: ArrayLike,
    dipole_positions: ArrayLike,
    corrected_positions: ArrayLike,
    *,
    n_bins: int = 6,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Mean / Median / P95 по бинам истинного расстояния (совместимо)."""
    true = np.asarray(true_positions, dtype=np.float64).reshape(-1, 3)
    base = np.asarray(dipole_positions, dtype=np.float64).reshape(-1, 3)
    corr = np.asarray(corrected_positions, dtype=np.float64).reshape(-1, 3)
    if n_bins < 1:
        raise ValueError("n_bins >= 1")
    keep = (
        np.all(np.isfinite(true), 1) & np.all(np.isfinite(base), 1) & np.all(np.isfinite(corr), 1)
    )
    true, base, corr = true[keep], base[keep], corr[keep]
    cols = [
        "bin",
        "r_left_m",
        "r_right_m",
        "n",
        "baseline_mean_m",
        "baseline_median_m",
        "baseline_p95_m",
        "corrected_mean_m",
        "corrected_median_m",
        "corrected_p95_m",
    ]
    if not len(true):
        return pd.DataFrame(columns=cols)
    r = np.linalg.norm(true, axis=1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        lo, hi = float(r.min()), float(r.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1e-9, n_bins + 1)
    else:
        raise ValueError("strategy: 'quantile' | 'uniform'")
    if len(edges) < 2:
        edges = np.array([float(r[0]) - 1e-9, float(r[0]) + 1e-9])
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    e_b = np.linalg.norm(true - base, axis=1)
    e_c = np.linalg.norm(true - corr, axis=1)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "r_left_m": float(edges[b]),
                "r_right_m": float(edges[b + 1]),
                "n": int(m.sum()),
                "baseline_mean_m": float(np.mean(e_b[m])),
                "baseline_median_m": float(np.median(e_b[m])),
                "baseline_p95_m": float(np.percentile(e_b[m], 95)),
                "corrected_mean_m": float(np.mean(e_c[m])),
                "corrected_median_m": float(np.median(e_c[m])),
                "corrected_p95_m": float(np.percentile(e_c[m], 95)),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 4. Ориентация дипольного момента (если есть)
# ──────────────────────────────────────────────


def orientation_metrics(
    true_moments: ArrayLike,
    estimated_moments: ArrayLike,
) -> dict[str, float]:
    """Угловая ошибка между истинным и оценённым дипольным моментом.

    Возвращает dict: mean_deg, median_deg, p95_deg, max_deg, n.
    """
    mt = np.asarray(true_moments, dtype=np.float64).reshape(-1, 3)
    me = np.asarray(estimated_moments, dtype=np.float64).reshape(-1, 3)
    if mt.shape != me.shape:
        raise ValueError("Формы true_moments и estimated_moments должны совпадать")
    keep = np.all(np.isfinite(mt), 1) & np.all(np.isfinite(me), 1)
    mt, me = mt[keep], me[keep]
    if not len(mt):
        return dict(n=0, mean_deg=np.nan, median_deg=np.nan, p95_deg=np.nan, max_deg=np.nan)
    nt = np.linalg.norm(mt, axis=1, keepdims=True)
    ne = np.linalg.norm(me, axis=1, keepdims=True)
    nt = np.where(nt == 0, 1, nt)
    ne = np.where(ne == 0, 1, ne)
    cos_a = np.clip(np.sum(mt / nt * me / ne, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_a))
    return dict(
        n=int(len(angles)),
        mean_deg=float(np.mean(angles)),
        median_deg=float(np.median(angles)),
        p95_deg=float(np.percentile(angles, 95)),
        max_deg=float(np.max(angles)),
    )


# ──────────────────────────────────────────────
# 5. Диагностика обучения (из history-CSV)
# ──────────────────────────────────────────────


def training_diagnostics(history: pd.DataFrame) -> dict[str, Any]:
    """Сходимость, переобучение, best-epoch.

    Ожидаемые колонки: epoch, train_loss, val_loss [, val_rmse, best_iteration].
    """
    out: dict[str, Any] = {}
    if history is None or history.empty:
        return out
    if "train_loss" in history and "val_loss" in history:
        tl = history["train_loss"].to_numpy(dtype=float)
        vl = history["val_loss"].to_numpy(dtype=float)
        finite = np.isfinite(tl) & np.isfinite(vl)
        if finite.any():
            gap = vl[finite] - tl[finite]
            out["final_train_loss"] = float(tl[np.isfinite(tl)][-1])
            out["final_val_loss"] = float(vl[np.isfinite(vl)][-1])
            out["final_gap"] = float(gap[-1])
            out["max_gap"] = float(np.max(gap))
            out["overfit_flag"] = bool(
                np.mean(gap[-5:]) > 0.15 * np.mean(np.abs(tl[np.isfinite(tl)][-5:]))
            )
    for col in ("val_rmse", "val_loss"):
        if col in history:
            v = history[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                out["best_val"] = float(np.min(v))
                out["best_epoch"] = int(np.argmin(history[col].to_numpy(dtype=float)))
                break
    if "best_iteration" in history:
        bi = history["best_iteration"].dropna()
        if len(bi):
            out["best_iteration"] = int(bi.iloc[-1])
    out["n_epochs"] = int(len(history))
    return out


# ──────────────────────────────────────────────
# 6. Статистика улучшения
# ──────────────────────────────────────────────


def improvement_stats(
    baseline_error: ArrayLike,
    corrected_error: ArrayLike,
    *,
    outlier_threshold_m: float = 0.5,
) -> dict[str, Any]:
    """Процентное улучшение, тест Уилкоксона, доля выбросов."""
    eb = _finite(np.asarray(baseline_error, dtype=np.float64).ravel())
    ec = _finite(np.asarray(corrected_error, dtype=np.float64).ravel())
    n = min(len(eb), len(ec))
    eb, ec = eb[:n], ec[:n]
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    out["baseline_mean"] = float(np.mean(eb))
    out["corrected_mean"] = float(np.mean(ec))
    out["improvement_mean_pct"] = (
        float(100.0 * (np.mean(eb) - np.mean(ec)) / np.mean(eb)) if np.mean(eb) > 0 else 0.0
    )
    out["improvement_median_pct"] = (
        float(100.0 * (np.median(eb) - np.median(ec)) / np.median(eb)) if np.median(eb) > 0 else 0.0
    )
    out["improvement_p95_pct"] = (
        float(100.0 * (np.percentile(eb, 95) - np.percentile(ec, 95)) / np.percentile(eb, 95))
        if np.percentile(eb, 95) > 0
        else 0.0
    )
    # Wilcoxon signed-rank (без зависимости от scipy — норм. аппроксимация)
    try:
        from scipy.stats import wilcoxon  # type: ignore

        d = eb - ec
        d = d[d != 0]
        if len(d) >= 10:
            stat, p = wilcoxon(d)
            out["wilcoxon_p"] = float(p)
            out["wilcoxon_significant"] = bool(p < 0.05)
        else:
            out["wilcoxon_p"] = np.nan
            out["wilcoxon_significant"] = None
    except ImportError:
        out["wilcoxon_p"] = np.nan
        out["wilcoxon_significant"] = None
    # Доля выбросов
    out["baseline_outlier_frac"] = float(np.mean(eb > outlier_threshold_m))
    out["corrected_outlier_frac"] = float(np.mean(ec > outlier_threshold_m))
    out["outlier_threshold_m"] = outlier_threshold_m
    return out


# ──────────────────────────────────────────────
# 7. Текстовая гистограмма (спарклайн в консоли)
# ──────────────────────────────────────────────


def _text_hist(values: np.ndarray, *, bins: int = 24, width: int = 40, title: str = "") -> str:
    """Компактная горизонтальная гистограмма для консоли."""
    v = _finite(values)
    if v.size == 0:
        return f"  {title}: нет данных"
    counts, edges = np.histogram(v, bins=bins)
    peak = counts.max() if counts.max() > 0 else 1
    lines = []
    if title:
        lines.append(f"  {title}")
    for i in range(bins):
        bar_len = int(round(counts[i] / peak * width))
        bar = "█" * bar_len
        lines.append(f"  {edges[i]:8.4f} │{bar} {counts[i]}")
    lines.append(f"           └{'─' * (width + 1)}")
    lines.append(f"  n={v.size}  range=[{v.min():.4f}, {v.max():.4f}]")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 8. ГЛАВНЫЙ КОНСОЛЬНЫЙ ОТЧЁТ
# ──────────────────────────────────────────────


def print_report(run_dir: str | Path, *, n_bins: int = 6) -> None:
    """Полный текстовый отчёт по каталогу запуска.

    Читает:
      - training_history.csv
      - test_predictions.csv   (колонки: r_true, baseline_error_m, corrected_error_m,
                                 и опционально *_x/y/z, true_moment_*, est_moment_*)
      - distance_bins.csv      (если уже посчитан; иначе строит из predictions)

    Выводит всё в stdout.
    """
    root = Path(run_dir)
    buf: list[str] = []
    _p = buf.append

    _p(_section("MAG-BEACON ML REPORT"))
    _p(f"  run directory : {root.resolve()}")
    _p(f"  generated     : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

    # ── 1. Кривая обучения ──────────────────
    hist_path = root / "training_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        diag = training_diagnostics(hist)
        _p(_section("1 │ TRAINING DIAGNOSTICS"))
        _p(f"  epochs / iterations : {diag.get('n_epochs', '–')}")
        if "best_iteration" in diag:
            _p(f"  best iteration      : {diag['best_iteration']}")
        elif "best_epoch" in diag:
            _p(f"  best epoch          : {diag['best_epoch']}")
        if "best_val" in diag:
            _p(f"  best val metric     : {diag['best_val']:.6f}")
        if "final_train_loss" in diag:
            _p(f"  final train loss    : {diag['final_train_loss']:.6f}")
            _p(f"  final val loss      : {diag['final_val_loss']:.6f}")
            _p(f"  final gap (V−T)     : {diag['final_gap']:+.6f}")
            _p(f"  max gap             : {diag['max_gap']:+.6f}")
            flag = diag.get("overfit_flag", False)
            _p(f"  overfit warning     : {'⚠  YES' if flag else 'no'}")

        # Мини-график сходимости (ASCII)
        if "train_loss" in hist and "val_loss" in hist:
            tl = hist["train_loss"].to_numpy(dtype=float)
            vl = hist["val_loss"].to_numpy(dtype=float)
            ok = np.isfinite(tl) & np.isfinite(vl)
            if ok.sum() >= 2:
                _p("")
                _p("  Loss curve (train · val):")
                step = max(1, ok.sum() // 20)
                idxs = list(range(0, len(tl), step))
                if idxs[-1] != len(tl) - 1:
                    idxs.append(len(tl) - 1)
                for i in idxs:
                    t = tl[i] if np.isfinite(tl[i]) else float("nan")
                    v = vl[i] if np.isfinite(vl[i]) else float("nan")
                    bar_t = "·" * int(min(t, 5) / 5 * 30) if np.isfinite(t) else ""
                    bar_v = "█" * int(min(v, 5) / 5 * 30) if np.isfinite(v) else ""
                    _p(f"    ep {i:4d}  T {t:.4f} {bar_t}")
                    _p(f"            V {v:.4f} {bar_v}")
    else:
        _p(_section("1 │ TRAINING DIAGNOSTICS"))
        _p("  training_history.csv не найден — пропускаю.")

    # ── 2. Тестовые предсказания ────────────
    pred_path = root / "test_predictions.csv"
    eb = ec = rr = None
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        _p(_section("2 │ TEST POSITION ERROR  (3-D Euclidean, m)"))

        if {"baseline_error_m", "corrected_error_m"}.issubset(pred.columns):
            eb = _finite(pred["baseline_error_m"].to_numpy(dtype=float))
            ec = _finite(pred["corrected_error_m"].to_numpy(dtype=float))
            sb, sc = _summary(eb), _summary(ec)
            _p(
                _table(
                    ["Estimator", "n", "Mean", "Median", "RMSE", "Std", "P90", "P95", "P99", "Max"],
                    [
                        [
                            "Baseline",
                            sb["n"],
                            sb["mean_m"],
                            sb["median_m"],
                            sb["rmse_m"],
                            sb["std_m"],
                            sb["p90_m"],
                            sb["p95_m"],
                            sb["p99_m"],
                            sb["max_m"],
                        ],
                        [
                            "Corrected",
                            sc["n"],
                            sc["mean_m"],
                            sc["median_m"],
                            sc["rmse_m"],
                            sc["std_m"],
                            sc["p90_m"],
                            sc["p95_m"],
                            sc["p99_m"],
                            sc["max_m"],
                        ],
                    ],
                )
            )

            # Улучшение
            _p(_section("3 │ IMPROVEMENT  (Baseline → Corrected)"))
            ist = improvement_stats(eb, ec)
            _p(f"  mean  improvement : {ist.get('improvement_mean_pct', 0):+.2f} %")
            _p(f"  median improvement: {ist.get('improvement_median_pct', 0):+.2f} %")
            _p(f"  P95   improvement : {ist.get('improvement_p95_pct', 0):+.2f} %")
            wp = ist.get("wilcoxon_p")
            if wp is not None and np.isfinite(wp):
                sig = ist.get("wilcoxon_significant")
                _p(
                    f"  Wilcoxon p-value  : {wp:.2e}  "
                    f"({'significant' if sig else 'n.s.'} at α=0.05)"
                )
            thr = ist.get("outlier_threshold_m", 0.5)
            _p(
                f"  outliers > {thr} m : baseline {ist.get('baseline_outlier_frac', 0) * 100:.1f}%  →  "
                f"corrected {ist.get('corrected_outlier_frac', 0) * 100:.1f}%"
            )

            # Гистограммы
            _p("")
            _p(_text_hist(eb, title="Baseline error distribution:"))
            _p("")
            _p(_text_hist(ec, title="Corrected error distribution:"))
        else:
            _p("  Колонки baseline_error_m / corrected_error_m не найдены.")

        # Расстояние
        if "r_true" in pred.columns:
            rr = _finite(pred["r_true"].to_numpy(dtype=float))

        # ── 3a. Покомпонентные ошибки ──────
        comp_cols = {ax: (f"baseline_err_{ax}", f"corrected_err_{ax}") for ax in "xyz"}
        if all(c in pred.columns for pair in comp_cols.values() for c in pair):
            _p(_section("3a│ PER-AXIS ERROR  (m)"))
            rows = []
            for ax in "xyz":
                db = _finite(pred[f"baseline_err_{ax}"].to_numpy(dtype=float))
                dc = _finite(pred[f"corrected_err_{ax}"].to_numpy(dtype=float))
                rows.append(
                    [
                        ax.upper(),
                        np.mean(np.abs(db)),
                        np.sqrt(np.mean(db**2)),
                        np.percentile(np.abs(db), 95),
                        np.mean(np.abs(dc)),
                        np.sqrt(np.mean(dc**2)),
                        np.percentile(np.abs(dc), 95),
                    ]
                )
            _p(
                _table(
                    ["Axis", "B-MAE", "B-RMSE", "B-P95", "C-MAE", "C-RMSE", "C-P95"],
                    rows,
                )
            )

        # ── 3b. Ориентация ─────────────────
        mom_cols = [c for c in pred.columns if "moment" in c.lower()]
        true_m = [c for c in mom_cols if c.startswith("true")]
        est_m = [c for c in mom_cols if c.startswith(("est", "pred"))]
        if len(true_m) == 3 and len(est_m) == 3:
            _p(_section("3b│ ORIENTATION ERROR  (dipole moment, deg)"))
            om = orientation_metrics(
                pred[sorted(true_m)].to_numpy(dtype=float),
                pred[sorted(est_m)].to_numpy(dtype=float),
            )
            _p(f"  n      : {om['n']}")
            _p(f"  mean   : {_fmt(om['mean_deg'], '°')}")
            _p(f"  median : {_fmt(om['median_deg'], '°')}")
            _p(f"  P95    : {_fmt(om['p95_deg'], '°')}")
            _p(f"  max    : {_fmt(om['max_deg'], '°')}")

    else:
        _p(_section("2 │ TEST PREDICTIONS"))
        _p("  test_predictions.csv не найден — пропускаю.")

    # ── 4. Биннинг по расстоянию ────────────
    _p(_section("4 │ ERROR vs DISTANCE  (bins)"))
    bins_path = root / "distance_bins.csv"
    bins_df = None
    if bins_path.exists():
        bins_df = pd.read_csv(bins_path)
    elif eb is not None and ec is not None and rr is not None:
        # пересчитать из предсказаний
        bins_df = _bins_from_errors(rr, eb, ec, n_bins=n_bins)

    if bins_df is not None and not bins_df.empty:
        rows = []
        for _, r in bins_df.iterrows():
            rows.append(
                [
                    int(r.get("bin", 0)),
                    f"{r.get('r_left_m', 0):.3f}–{r.get('r_right_m', 0):.3f}",
                    int(r.get("n", 0)),
                    r.get("baseline_mean_m", np.nan),
                    r.get("baseline_p95_m", np.nan),
                    r.get("corrected_mean_m", np.nan),
                    r.get("corrected_p95_m", np.nan),
                ]
            )
        _p(
            _table(
                ["Bin", "r range (m)", "n", "B-mean", "B-P95", "C-mean", "C-P95"],
                rows,
            )
        )
        # ASCII-график error vs r
        _p("")
        _p("  Corrected mean error vs distance:")
        rm = 0.5 * (bins_df["r_left_m"].to_numpy() + bins_df["r_right_m"].to_numpy())
        cm = bins_df["corrected_mean_m"].to_numpy(dtype=float)
        bm = bins_df["baseline_mean_m"].to_numpy(dtype=float)
        mx = max(np.nanmax(cm), np.nanmax(bm)) if len(cm) else 1
        for i in range(len(rm)):
            bc = "█" * int(np.nan_to_num(cm[i]) / mx * 35) if np.isfinite(cm[i]) else ""
            bb = "░" * int(np.nan_to_num(bm[i]) / mx * 35) if np.isfinite(bm[i]) else ""
            _p(f"    r={rm[i]:6.3f}  B {bb}")
            _p(f"              C {bc}")
    else:
        _p("  Нет данных для бинов.")

    # ── 5. Итог ─────────────────────────────
    _p(_section("5 │ SUMMARY"))
    if eb is not None and ec is not None:
        sb, sc = _summary(eb), _summary(ec)
        _p(f"  Baseline  RMSE = {_fmt(sb['rmse_m'])}   Corrected RMSE = {_fmt(sc['rmse_m'])}")
        _p(f"  Baseline  P95  = {_fmt(sb['p95_m'])}   Corrected  P95  = {_fmt(sc['p95_m'])}")
        imp = 100 * (sb["rmse_m"] - sc["rmse_m"]) / sb["rmse_m"] if sb["rmse_m"] > 0 else 0
        _p(f"  RMSE improvement : {imp:+.2f} %")
    _p(_DLINE)
    _p("")

    print("\n".join(buf))


# ──────────────────────────────────────────────
# Вспомогательное: бины из векторов ошибок
# ──────────────────────────────────────────────


def _bins_from_errors(
    r: np.ndarray,
    eb: np.ndarray,
    ec: np.ndarray,
    *,
    n_bins: int = 6,
) -> pd.DataFrame:
    keep = np.isfinite(r) & np.isfinite(eb) & np.isfinite(ec)
    r, eb, ec = r[keep], eb[keep], ec[keep]
    if not len(r):
        return pd.DataFrame()
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([r[0] - 1e-9, r[0] + 1e-9])
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "r_left_m": edges[b],
                "r_right_m": edges[b + 1],
                "n": int(m.sum()),
                "baseline_mean_m": float(np.mean(eb[m])),
                "baseline_median_m": float(np.median(eb[m])),
                "baseline_p95_m": float(np.percentile(eb[m], 95)),
                "corrected_mean_m": float(np.mean(ec[m])),
                "corrected_median_m": float(np.median(ec[m])),
                "corrected_p95_m": float(np.percentile(ec[m], 95)),
            }
        )
    return pd.DataFrame(rows)
