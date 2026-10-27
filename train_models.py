"""
5단계: 라인별 TCI 모델 비교

입력 : build_dataset.py 결과 (dataset/lane_*.csv.gz)
출력 : results/metrics.csv           라인 x 모델 성능 (test)
       results/feature_importance.csv 선형 계수, 트리 계열 중요도
       results/pred_<LANE>.csv.gz     test 예측값(=ΔTCI) - 6·7단계 해석/활용용

모델 (같은 X, 같은 Y, 같은 분할로 비교)
  Baseline_CS : CS 격차 하나만 쓰는 선형회귀 (골드 공식 재현 수준의 기준선)
  Linear      : 다중 선형회귀
  Ridge/Lasso : 정규화 선형회귀, alpha는 train 5-fold(게임 단위)로 선택
  DecisionTree, RandomForest, XGBoost, LightGBM : 하이퍼파라미터는 fold 0을 검증용으로 선택
  MLP (--mlp 옵션) : 기본 인공신경망

평가 지표 (test 게임에서 한 번만 계산)
  RMSE, MAE, R2 : 예측 Y vs 실제 Y
  SignAcc_gold  : 예측 Y 부호 vs 실제 골드 격차 부호 (|골드 격차| >= 100, 논문 방식)

사용 예
  python train_models.py --data_dir ./dataset --out_dir ./results
  python train_models.py --data_dir ./dataset --out_dir ./results_quick --max_train_games 5000   # 빠른 시험
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

SEED = 42
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")  # LightGBM 버전별 안내 메시지
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
CS_COL = {"TOP": "z_minionsKilled", "JUNGLE": "z_jungleMinionsKilled",
          "MIDDLE": "z_minionsKilled", "BOTTOM": "z_adc_minionsKilled"}


def metrics(y, p, gold):
    m = np.abs(gold) >= 100
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "R2": r2_score(y, p), "SignAcc_gold": np.mean((p[m] > 0) == (gold[m] > 0))}


def pick_by_val(make, grid, Xtr, ytr, val_mask):
    """fold 0(게임 단위)을 검증용으로 두고 RMSE가 가장 낮은 설정 선택."""
    best, best_rmse = None, np.inf
    for params in grid:
        m = make(**params).fit(Xtr[~val_mask], ytr[~val_mask])
        rmse = np.sqrt(mean_squared_error(ytr[val_mask], m.predict(Xtr[val_mask])))
        if rmse < best_rmse:
            best, best_rmse = params, rmse
    return best


def run_lane(lane, path, args, rows, imps):
    d = pd.read_csv(path)
    feats = [c for c in d.columns if c.startswith("z_")]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if args.max_train_games:
        g = tr.matchId.drop_duplicates().sample(n=min(args.max_train_games, tr.matchId.nunique()),
                                                random_state=SEED)
        tr = tr[tr.matchId.isin(g)]
    Xtr, ytr = tr[feats].to_numpy(np.float32), tr.Y.to_numpy()
    Xte, yte, gold_te = te[feats].to_numpy(np.float32), te.Y.to_numpy(), te.gold_diff.to_numpy()
    fold = tr.fold.to_numpy()
    val = fold == 0
    print(f"\n== {lane}: train {tr.matchId.nunique()}게임 {len(tr):,}행 / test {te.matchId.nunique()}게임, X {len(feats)}개")

    preds = te[["matchId", "minute", "Y", "gold_diff", "blueWin"]].copy()

    def record(name, model, Xa=Xtr, Xb=Xte, note=""):
        t0 = time.time()
        model.fit(Xa, ytr)
        p = model.predict(Xb)
        r = {"lane": lane, "model": name, **metrics(yte, p, gold_te),
             "fit_sec": round(time.time() - t0, 1), "params": note}
        rows.append(r)
        preds[name] = p
        print(f"  {name:13s} RMSE {r['RMSE']:.3f}  MAE {r['MAE']:.3f}  R2 {r['R2']:.3f}  "
              f"SignAcc {r['SignAcc_gold']:.3f}  ({r['fit_sec']}s) {note}")
        return model

    # 1) 기준선: CS 격차만
    cs_i = feats.index(CS_COL[lane])
    record("Baseline_CS", LinearRegression(), Xtr[:, [cs_i]], Xte[:, [cs_i]])

    # 2) 선형 계열
    lin = record("Linear", LinearRegression())
    ps = PredefinedSplit(fold)  # 게임 단위 5-fold
    alphas = np.logspace(-3, 3, 13)
    rid = record("Ridge", RidgeCV(alphas=alphas, cv=ps))
    las = record("Lasso", LassoCV(alphas=np.logspace(-4, 0, 9), cv=ps, random_state=SEED, max_iter=5000))
    for name, m in [("Linear", lin), ("Ridge", rid), ("Lasso", las)]:
        for f, w in zip(feats, m.coef_):
            imps.append([lane, name, f, float(w)])

    # 3) 트리 계열 (fold 0 검증으로 설정 선택)
    dt_grid = [{"max_depth": md, "min_samples_leaf": 50, "random_state": SEED} for md in [4, 6, 8, 10, 12]]
    p = pick_by_val(DecisionTreeRegressor, dt_grid, Xtr, ytr, val)
    record("DecisionTree", DecisionTreeRegressor(**p), note=f"max_depth={p['max_depth']}")

    rf_grid = [{"n_estimators": 100, "max_depth": md, "min_samples_leaf": 50, "max_features": 0.5,
                "n_jobs": -1, "random_state": SEED} for md in [8, 12, 16]]
    p = pick_by_val(RandomForestRegressor, rf_grid, Xtr, ytr, val)
    rf = record("RandomForest", RandomForestRegressor(**p), note=f"max_depth={p['max_depth']}")
    for f, w in zip(feats, rf.feature_importances_):
        imps.append([lane, "RandomForest", f, float(w)])

    # 4) 부스팅 계열: fold 0 검증으로 조기 종료 → 그 반복 수로 전체 train 재학습
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
        m.fit(Xtr[~val], ytr[~val], eval_set=[(Xtr[val], ytr[val])], verbose=False)
        n = m.best_iteration + 1
        xm = record("XGBoost", xgb.XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=6,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=SEED, n_jobs=-1), note=f"n_estimators={n}")
        for f, w in zip(feats, xm.feature_importances_):
            imps.append([lane, "XGBoost", f, float(w)])
    except ImportError:
        print("  XGBoost 미설치: pip install xgboost")

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(Xtr[~val], ytr[~val], eval_set=[(Xtr[val], ytr[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        n = m.best_iteration_
        lm = record("LightGBM", lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=63,
                                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                  random_state=SEED, verbose=-1), note=f"n_estimators={n}")
        gain = lm.booster_.feature_importance(importance_type="gain")
        for f, w in zip(feats, gain / gain.sum()):
            imps.append([lane, "LightGBM", f, float(w)])
    except ImportError:
        print("  LightGBM 미설치: pip install lightgbm")

    # 5) (선택) 인공신경망
    if args.mlp:
        record("MLP", MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                   random_state=SEED, max_iter=200))

    preds.to_csv(os.path.join(args.out_dir, f"pred_{lane}.csv.gz"), index=False, compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lanes", nargs="+", default=LANES)
    ap.add_argument("--max_train_games", type=int, default=0, help="0이면 train 전체 사용")
    ap.add_argument("--mlp", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, imps = [], []
    for lane in args.lanes:
        run_lane(lane, os.path.join(args.data_dir, f"lane_{lane}.csv.gz"), args, rows, imps)
        # 라인마다 중간 저장 (도중에 멈춰도 결과 보존)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
        pd.DataFrame(imps, columns=["lane", "model", "feature", "value"]) \
            .to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False)
    print("\n완료:", args.out_dir)


if __name__ == "__main__":
    main()
