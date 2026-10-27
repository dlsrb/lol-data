"""
제외한 후보 피처(soloDeaths, objPart)의 근거 - 전체 데이터 버전

README 2.2절의 "soloDeaths는 솔로킬과 상관 −1", "objPart는 타겟과 상관이 거의 없고 objKA와 중복"을
전체 데이터(정제 후 36,527게임, 5~40분)로 계산한다.

입력 : csv_v2/games.csv.gz, csv_v2/player_minute.csv.gz
출력 : results/feature_screen.csv

사용 예
  python feature_screen.py --csv_dir ./csv_v2 --results_dir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd

import build_dataset as bd  # 같은 정제·피처 계산 규칙을 그대로 사용

bd.USECOLS = bd.USECOLS + ["soloDeaths", "teamObj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    keep, pm = bd.load(args.csv_dir, "")
    pm = bd.add_player_features(pm)
    pm["objPart"] = (pm.objKA / pm.teamObj.replace(0, np.nan)).fillna(0)

    cols = ["totalGold", "xp", "soloKills", "soloDeaths", "objKA", "objPart"]
    rows = []
    for lane in ["TOP", "JUNGLE", "MIDDLE"]:
        s = pm[pm.teamPosition == lane]
        b = s[s.teamId == 100].set_index(["matchId", "minute"])[cols]
        r = s[s.teamId == 200].set_index(["matchId", "minute"])[cols]
        d = (b - r).dropna()
        g = d.groupby(level="minute")
        z = (d - g.transform("mean")) / g.transform("std").replace(0, 1)
        y = z.totalGold + z.xp
        rows += [
            [lane, "corr(soloKills, soloDeaths)", z.soloKills.corr(z.soloDeaths)],
            [lane, "corr(objPart, Y)", z.objPart.corr(y)],
            [lane, "corr(objKA, Y)", z.objKA.corr(y)],
            [lane, "corr(objPart, objKA)", z.objPart.corr(z.objKA)],
        ]
    res = pd.DataFrame(rows, columns=["lane", "metric", "value"])
    res.to_csv(os.path.join(args.results_dir, "feature_screen.csv"), index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
