"""
보완 1차 - 5번: 탐색적 데이터 분석(EDA)

목적: 보고서에 나오는 지표(골드+XP 격차, 피처, MCI)와 확률(승률, 부호 일치율)이
      데이터에서 어떻게 생겨나는지 그림으로 설명한다.

입력 : dataset/lane_*.csv.gz, results/tci_coefficients.csv
출력 : images/eda1_spread_by_minute.png   골드 격차의 퍼짐이 시간에 따라 커짐 → 분별 z-score가 필요한 이유
       images/eda2_winprob_by_target.png  골드+XP 격차(Y)에 따른 승리 확률 → [Table 1]의 의미
       images/eda3_feature_corr.png       라인별 피처와 Y의 상관 → 피처 선정 근거
       images/eda4_winprob_by_mci.png     MCI에 따른 승리 확률 → [Table 8]의 연속 버전
       results/eda_summary.csv            그림에 쓰인 수치

사용 예
  python eda.py --data_dir ./dataset --results_dir ./results --img_dir ./images
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
COLORS = {"TOP": "tab:red", "JUNGLE": "tab:green", "MIDDLE": "tab:blue", "BOTTOM": "tab:orange",
          "ADC": "tab:orange", "SPT": "tab:purple"}


def load(data_dir):
    out = {}
    for lane in LANES:
        out[lane] = pd.read_csv(os.path.join(data_dir, f"lane_{lane}.csv.gz"))
    return out


def eda1_spread(data, path, rows):
    """분 시점별 골드 격차의 표준편차: 시간이 지날수록 격차의 크기 자체가 커진다."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for lane, d in data.items():
        s = d.groupby("minute").gold_diff.std()
        ax.plot(s.index, s.values, label=lane, color=COLORS[lane])
        for m in (5, 15, 25, 35):
            rows.append(["eda1", lane, f"gold_diff_std@{m}", float(s.get(m, np.nan))])
    ax.set_xlabel("minute")
    ax.set_ylabel("std of gold difference (blue - red)")
    ax.set_title("Spread of lane gold difference grows over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def eda2_winprob(data, path, rows, minutes=(10, 15, 20)):
    """Y(골드+XP 격차) 구간별 블루팀 승리 확률."""
    fig, axes = plt.subplots(1, len(minutes), figsize=(15, 4), sharey=True)
    bins = np.arange(-5, 5.5, 0.5)
    for ax, m in zip(axes, minutes):
        for lane, d in data.items():
            s = d[d.minute == m]
            b = pd.cut(s.Y, bins)
            p = s.groupby(b, observed=True).blueWin.agg(["mean", "size"])
            p = p[p["size"] >= 50]  # 표본이 적은 구간은 제외
            x = [iv.mid for iv in p.index]
            ax.plot(x, p["mean"], marker="o", ms=3, label=lane, color=COLORS[lane])
            acc = ((s.Y > 0) == (s.blueWin == 1)).mean()
            rows.append(["eda2", lane, f"sign_acc@{m}", float(acc)])
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{m} min")
        ax.set_xlabel("Y = z(gold diff) + z(XP diff)")
    axes[0].set_ylabel("P(blue team wins)")
    axes[0].legend()
    fig.suptitle("Win probability by lane gold+XP advantage")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def eda3_corr(data, path, rows):
    """라인별 피처(격차, z-score)와 Y의 상관."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, lane in zip(axes, LANES):
        d = data[lane]
        feats = [c for c in d.columns if c.startswith("z_")]
        c = d[feats + ["Y"]].corr()["Y"].drop("Y").sort_values()
        ax.barh([f[2:] for f in c.index], c.values,
                color=["tab:red" if v < 0 else "tab:blue" for v in c.values])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(lane)
        for f, v in c.items():
            rows.append(["eda3", lane, f"corr_Y:{f[2:]}", float(v)])
    axes[0].set_xlabel("correlation with Y")
    fig.suptitle("Correlation between each feature and the target (all minutes, all games)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def eda4_mci(data, coef, path, rows):
    """게임 단위 MCI(블루 기준)에 따른 블루팀 승리 확률. BOTTOM은 ADC/SPT 항 분리."""
    parts = {"TOP": ("TOP", None), "JUNGLE": ("JUNGLE", None), "MIDDLE": ("MIDDLE", None),
             "ADC": ("BOTTOM", "adc_"), "SPT": ("BOTTOM", "spt_")}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(-3, 3.25, 0.25)
    for role, (lane, pre) in parts.items():
        d = data[lane]
        w = coef[coef.lane == lane].set_index("feature").coef
        feats = [f for f in w.index if pre is None or f.startswith(pre)]
        d = d.assign(tci=d[["z_" + f for f in feats]].to_numpy() @ w[feats].to_numpy())
        g = d.groupby("matchId").agg(score=("tci", "mean"), win=("blueWin", "first"))
        # 블루 플레이어(점수 그대로)와 레드 플레이어(부호 반전, 승패 반전)를 모두 포함 → [Table 8]과 같은 기준
        g = pd.concat([g, pd.DataFrame({"score": -g.score, "win": 1 - g.win})], ignore_index=True)
        g["MCI"] = g.score / g.score.std()
        p = g.groupby(pd.cut(g.MCI, bins), observed=True).win.agg(["mean", "size"])
        p = p[p["size"] >= 30]
        ax.plot([iv.mid for iv in p.index], p["mean"], marker="o", ms=3, label=role, color=COLORS[role])
        rows.append(["eda4", role, "P(win|MCI>=1)", float(g[g.MCI >= 1].win.mean())])
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("MCI (game-level TCI)")
    ax.set_ylabel("P(player's team wins)")
    ax.set_title("Win probability by match contribution index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    data = load(args.data_dir)
    coef = pd.read_csv(os.path.join(args.results_dir, "tci_coefficients.csv"))
    rows = []
    eda1_spread(data, os.path.join(args.img_dir, "eda1_spread_by_minute.png"), rows)
    eda2_winprob(data, os.path.join(args.img_dir, "eda2_winprob_by_target.png"), rows)
    eda3_corr(data, os.path.join(args.img_dir, "eda3_feature_corr.png"), rows)
    eda4_mci(data, coef, os.path.join(args.img_dir, "eda4_winprob_by_mci.png"), rows)
    pd.DataFrame(rows, columns=["figure", "group", "metric", "value"]) \
        .to_csv(os.path.join(args.results_dir, "eda_summary.csv"), index=False)
    print("완료")


if __name__ == "__main__":
    main()
