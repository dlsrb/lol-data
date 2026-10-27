"""
보완 1차 - 4번: 대표 모델 2개(Ridge vs XGBoost) 비교와 시각화

선정 이유
  Ridge   : 선형 계열 대표. 가중치를 그대로 해석할 수 있음
  XGBoost : 트리·부스팅 계열 대표. 8개 모델 중 성능이 가장 높음(LightGBM과 동률)
  Baseline_CS(CS만 사용)는 비교 기준선으로 함께 표시

입력 : results/pred_<LANE>.csv.gz (train_models.py 결과, 평가 데이터 예측값)
출력 : images/cmp1_overall.png     라인별 R², RMSE 막대그래프
       images/cmp2_by_minute.png   분 시점별 RMSE 변화
       images/cmp3_pred_vs_true.png 예측값 vs 실제값 (밀도 그림)
       results/compare_summary.csv 그림에 쓰인 수치 + 게임 단위 부트스트랩 신뢰구간

사용 예
  python compare_models.py --results_dir ./results --img_dir ./images
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
MODELS = {"Baseline_CS": "lightgray", "Ridge": "tab:blue", "XGBoost": "tab:orange"}
SEED = 42


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def bootstrap_diff(d, n_boot=500):
    """XGBoost R² - Ridge R²의 95% 신뢰구간. 같은 게임의 분들은 함께 뽑는다."""
    rng = np.random.default_rng(SEED)
    g = d.assign(er=(d.Y - d.Ridge) ** 2, ex=(d.Y - d.XGBoost) ** 2,
                 yy=(d.Y - d.Y.mean()) ** 2).groupby("matchId")[["er", "ex", "yy"]].sum().to_numpy()
    diffs = []
    for _ in range(n_boot):
        s = g[rng.integers(0, len(g), len(g))].sum(axis=0)
        diffs.append((s[0] - s[1]) / s[2])
    return np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    preds = {l: pd.read_csv(os.path.join(args.results_dir, f"pred_{l}.csv.gz"),
                            usecols=["matchId", "minute", "Y"] + list(MODELS)) for l in LANES}
    rows = []

    # 1) 전체 성능
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(LANES))
    for k, (name, col) in enumerate(MODELS.items()):
        r2s = [r2(preds[l].Y, preds[l][name]) for l in LANES]
        rms = [rmse(preds[l].Y, preds[l][name]) for l in LANES]
        axes[0].bar(x + (k - 1) * 0.27, r2s, 0.27, label=name, color=col)
        axes[1].bar(x + (k - 1) * 0.27, rms, 0.27, label=name, color=col)
        for l, a, b in zip(LANES, r2s, rms):
            rows.append([l, name, "all", a, b])
    for ax, t in zip(axes, ["R² (higher is better)", "RMSE (lower is better)"]):
        ax.set_xticks(x, LANES)
        ax.set_title(t)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.img_dir, "cmp1_overall.png"), dpi=120)

    # 2) 분 시점별 RMSE
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, l in zip(axes, LANES):
        d = preds[l]
        for name in ["Ridge", "XGBoost"]:
            s = d.groupby("minute").apply(lambda g: rmse(g.Y, g[name]), include_groups=False)
            ax.plot(s.index, s.values, label=name, color=MODELS[name])
            for m in (10, 20, 30):
                rows.append([l, name, f"min{m}", np.nan, float(s.get(m, np.nan))])
        ax.set_title(l)
        ax.set_xlabel("minute")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    fig.suptitle("Prediction error by game minute (test games)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.img_dir, "cmp2_by_minute.png"), dpi=120)

    # 3) 예측값 vs 실제값
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for j, l in enumerate(LANES):
        d = preds[l]
        for i, name in enumerate(["Ridge", "XGBoost"]):
            ax = axes[i, j]
            ax.hexbin(d.Y, d[name], gridsize=60, bins="log", cmap="Blues", extent=(-8, 8, -8, 8))
            ax.plot([-8, 8], [-8, 8], color="red", lw=0.8)
            ax.set_title(f"{l} - {name} (R²={r2(d.Y, d[name]):.3f})")
            if i == 1:
                ax.set_xlabel("actual Y")
            if j == 0:
                ax.set_ylabel("predicted Y (TCI)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.img_dir, "cmp3_pred_vs_true.png"), dpi=110)

    # 4) 성능 차이의 신뢰구간
    for l in LANES:
        lo, hi = bootstrap_diff(preds[l])
        rows.append([l, "XGB-Ridge", "R2_diff_CI95", lo, hi])
        print(f"{l}: XGBoost R² - Ridge R² 95% 신뢰구간 [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame(rows, columns=["lane", "model", "scope", "R2_or_low", "RMSE_or_high"]) \
        .to_csv(os.path.join(args.results_dir, "compare_summary.csv"), index=False)
    print("완료")


if __name__ == "__main__":
    main()
