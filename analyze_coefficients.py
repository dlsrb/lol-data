"""
6단계: 최종 TCI(Ridge) 계수 해석

입력 : dataset/lane_*.csv.gz
출력 : results/tci_coefficients.csv  라인별 계수, 95% 신뢰구간(게임 단위 부트스트랩), VIF, 단순상관
       results/tci_coefficients.png  라인별 계수 막대그래프 (README 삽입용)

사용 예
  python analyze_coefficients.py --data_dir ./dataset --out_dir ./results --n_boot 100
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import PredefinedSplit

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
SEED = 42


def analyze(lane, path, n_boot):
    d = pd.read_csv(path)
    feats = [c for c in d.columns if c.startswith("z_")]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    X, y = tr[feats].to_numpy(), tr.Y.to_numpy()

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=PredefinedSplit(tr.fold.to_numpy())).fit(X, y)
    r2 = r2_score(te.Y, ridge.predict(te[feats].to_numpy()))

    # 다중공선성 진단: VIF = 상관행렬 역행렬의 대각원소 (5 이상이면 주의, 10 이상이면 심각)
    vif = np.diag(np.linalg.inv(np.corrcoef(X.T)))

    # 계수 안정성: 게임 단위로 다시 뽑아 재학습 (같은 게임의 분들은 함께 뽑힘)
    rng = np.random.default_rng(SEED)
    idx = tr.groupby("matchId").indices
    games = np.array(list(idx.keys()))
    boots = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in games[rng.integers(0, len(games), len(games))]])
        boots.append(LinearRegression().fit(X[ii], y[ii]).coef_)
    boots = np.array(boots)

    out = pd.DataFrame({
        "lane": lane, "feature": [f[2:] for f in feats], "coef": ridge.coef_,
        "ci_low": np.percentile(boots, 2.5, axis=0), "ci_high": np.percentile(boots, 97.5, axis=0),
        "VIF": vif, "corr_with_Y": [np.corrcoef(X[:, i], y)[0, 1] for i in range(len(feats))],
    })
    out["sign_flip"] = np.sign(out.coef) != np.sign(out.corr_with_Y)  # 단순상관과 부호가 반대인 변수
    print(f"{lane}: alpha={ridge.alpha_:g}, intercept={ridge.intercept_:.4f}, test R2={r2:.4f}")
    return out.reindex(out.coef.abs().sort_values(ascending=False).index)


def plot(all_coef, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True)
    for ax, lane in zip(axes, LANES):
        c = all_coef[all_coef.lane == lane].iloc[::-1]
        colors = ["tab:blue" if v > 0 else "tab:red" for v in c.coef]
        ax.barh(c.feature, c.coef, color=colors,
                xerr=[c.coef - c.ci_low, c.ci_high - c.coef], capsize=2)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(lane)
    axes[0].set_xlabel("Ridge coefficient (z-scored X)")
    fig.suptitle("TCI weights by lane (Y = z(gold diff) + z(XP diff))")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    res = pd.concat([analyze(l, os.path.join(args.data_dir, f"lane_{l}.csv.gz"), args.n_boot)
                     for l in LANES], ignore_index=True)
    res.to_csv(os.path.join(args.out_dir, "tci_coefficients.csv"), index=False)
    plot(res, os.path.join(args.out_dir, "tci_coefficients.png"))
    print("완료:", args.out_dir)


if __name__ == "__main__":
    main()
