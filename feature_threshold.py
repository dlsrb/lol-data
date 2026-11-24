"""
피처 선택 실험: 타겟과의 상관 절대값 0.5 이상인 피처만 쓰면 결과가 더 좋은가?

- 상관은 학습 데이터로만 계산해 피처를 고른다(평가 데이터 정보 사용 안 함).
- 같은 릿지 회귀, 같은 학습/평가 분할로 '전체 피처'와 '|상관| >= 0.5 피처만'을 비교한다.
입력 : dataset/lane_*.csv.gz
출력 : results/feature_threshold.csv
사용 예
  python feature_threshold.py --data_dir ./dataset --results_dir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]


def fit(tr, te, feats):
    m = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=PredefinedSplit(tr.fold.to_numpy())).fit(tr[feats], tr.Y)
    p = m.predict(te[feats])
    sign = (np.sign(p) == np.sign(te.gold_diff)) [te.gold_diff.abs() >= 100].mean()
    return r2_score(te.Y, p), mean_squared_error(te.Y, p) ** 0.5, sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    rows = []
    for lane in LANES:
        d = pd.read_csv(os.path.join(a.data_dir, f"lane_{lane}.csv.gz"))
        tr, te = d[d.split == "train"], d[d.split == "test"]
        feats = [c for c in d.columns if c.startswith("z_")]
        corr = tr[feats].corrwith(tr.Y)
        strong = [f for f in feats if abs(corr[f]) >= a.threshold]
        dropped = [f[2:] for f in feats if f not in strong]
        for name, fs in (("전체 피처", feats), (f"|상관|>={a.threshold} 피처만", strong)):
            r2, rmse, sign = fit(tr, te, fs)
            rows.append([lane, name, len(fs), round(r2, 4), round(rmse, 4), round(sign, 4),
                         "" if fs is feats else ", ".join(dropped)])
            print(rows[-1])
    pd.DataFrame(rows, columns=["lane", "setting", "n_features", "R2", "RMSE", "SignAcc_gold", "dropped"]) \
        .to_csv(os.path.join(a.results_dir, "feature_threshold.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
