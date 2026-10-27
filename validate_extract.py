"""
추출 데이터 검증 - 전체 데이터 버전 (README 1.1절 수치)

타임라인 마지막 프레임의 누적값이 매치 데이터의 종료 시점 기록과 얼마나 일치하는지 계산한다.
  - 골드, 킬, 데스, 어시스트, 와드 설치 일치율
  - CC 시간(timeEnemySpentControlled, ms)과 매치 데이터 totalTimeCCDealt의 상관

입력 : csv_v2/player_minute.csv.gz, csv_v2/player_end.csv.gz  (정제 전 전체 38,075게임)
출력 : results/extract_validation.csv

사용 예
  python validate_extract.py --csv_dir ./csv_v2 --results_dir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    use = ["matchId", "participantId", "minute", "totalGold", "kills", "deaths", "assists",
           "wardsPlaced", "ccEnemy_ms"]
    last = None
    for ch in pd.read_csv(os.path.join(args.csv_dir, "player_minute.csv.gz"), usecols=use, chunksize=1_000_000):
        # 청크마다 (게임, 참가자)별 가장 늦은 분만 남기고, 청크 경계는 아래에서 다시 정리
        ch = ch.sort_values("minute").groupby(["matchId", "participantId"]).tail(1)
        last = ch if last is None else pd.concat([last, ch])
    last = last.sort_values("minute").groupby(["matchId", "participantId"]).tail(1)

    end = pd.read_csv(os.path.join(args.csv_dir, "player_end.csv.gz"))
    m = last.merge(end, on=["matchId", "participantId"], suffixes=("", "_end"))

    rows = []
    for a, b in [("totalGold", "goldEarned"), ("kills", "kills_end"), ("deaths", "deaths_end"),
                 ("assists", "assists_end"), ("wardsPlaced", "wardsPlaced_end")]:
        rows.append([f"match rate: {a} vs {b}", float((m[a] == m[b]).mean())])
    rows.append(["corr: ccEnemy_ms vs totalTimeCCDealt", float(np.corrcoef(m.ccEnemy_ms, m.totalTimeCCDealt)[0, 1])])
    rows.append(["median ratio: ccEnemy_ms / totalTimeCCDealt",
                 float((m.ccEnemy_ms / m.totalTimeCCDealt.replace(0, np.nan)).median())])

    rows.append(["players", len(m)])

    res = pd.DataFrame(rows, columns=["metric", "value"])
    res.to_csv(os.path.join(args.results_dir, "extract_validation.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
