"""
결론(5장) 근거: 라인별 핵심 플레이의 '게임 종료 시점 맞상대와의 차이'에 따른 승률과 평균 MCI

입력 : csv_v2/games.csv.gz, csv_v2/player_minute.csv.gz, results/mci_games.csv.gz
출력 : results/conclusion_check.csv   (역할, 플레이, 맞상대와의 차이, 게임 수, 승률(%), 평균 MCI)

차이 = 마지막 분석 시점(종료 프레임 직전, 40분 이하)의 누적값, 내 값 − 맞상대 값
사용 예
  python conclusion_check.py --csv_dir ./csv_v2 --results_dir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd

CHECKS = [("TOP", "plates"), ("JUNGLE", "jungleMinionsKilled"), ("JUNGLE", "gank"),
          ("MIDDLE", "minionsKilled"), ("ADC", "deaths"), ("SPT", "roam")]
POS2ROLE = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "BOTTOM": "ADC", "UTILITY": "SPT"}
OWN_ZONE = {"TOP": "ka_TOP", "MIDDLE": "ka_MID", "BOTTOM": "ka_BOT", "UTILITY": "ka_BOT"}
ZONES = ["ka_TOP", "ka_MID", "ka_BOT", "ka_JUNGLE"]
SMALL = ([-np.inf, -2.5, -0.5, 0.5, 2.5, np.inf], ["-3 이하", "-2~-1", "0", "+1~+2", "+3 이상"])
LARGE = ([-np.inf, -20.5, -0.5, 0.5, 20.5, np.inf], ["-21 이하", "-20~-1", "0", "+1~+20", "+21 이상"])


FEATURE_KR = {"plates": "포탑 방패", "jungleMinionsKilled": "정글 CS", "gank": "갱킹",
              "minionsKilled": "CS", "deaths": "데스", "roam": "로밍"}


def pretty(t):
    """사람이 읽기 쉬운 표: 한글 열 이름, 승률은 %, 소수점 정리."""
    return pd.DataFrame({
        "역할": t.role,
        "플레이": t.feature.map(FEATURE_KR),
        "맞상대와의 차이": t.bin.astype(str),
        "게임 수": t.games,
        "승률(%)": (t.win_rate * 100).round(1),
        "평균 MCI": t.MCI_mean.round(2),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    g = pd.read_csv(os.path.join(args.csv_dir, "games.csv.gz"))
    keep = set(g[(g.queueId == 420) & (~g.gameVersion.astype(str).str.startswith("14"))
                 & (g.gameDuration_s <= 2400)].matchId)
    cols = ["matchId", "minute", "teamId", "teamPosition", "plates", "minionsKilled",
            "jungleMinionsKilled", "deaths"] + ZONES
    pm = pd.concat([ch[ch.matchId.isin(keep)] for ch in
                    pd.read_csv(os.path.join(args.csv_dir, "player_minute.csv.gz"), usecols=cols,
                                chunksize=2_000_000)])
    last = pm.groupby("matchId").minute.transform("max")
    pm = pm[(pm.minute < last) & (pm.minute <= 40)]
    end = pm.loc[pm.groupby(["matchId", "teamPosition", "teamId"]).minute.idxmax()].copy()
    end["gank"] = end.ka_TOP + end.ka_MID + end.ka_BOT
    end["roam"] = np.nan
    for pos, own in OWN_ZONE.items():
        m = end.teamPosition == pos
        end.loc[m, "roam"] = end.loc[m, ZONES].sum(axis=1) - end.loc[m, own]
    end["role"] = end.teamPosition.map(POS2ROLE)

    mci = pd.read_csv(os.path.join(args.results_dir, "mci_games.csv.gz"))
    x = end.merge(mci[["matchId", "role", "teamId", "win", "MCI"]], on=["matchId", "role", "teamId"])

    rows = []
    for role, feat in CHECKS:
        s = x[x.role == role][["matchId", "teamId", feat, "win", "MCI"]]
        o = s[["matchId", "teamId", feat]].assign(teamId=lambda d: 300 - d.teamId)
        s = s.merge(o, on=["matchId", "teamId"], suffixes=("", "_opp"))
        edges, labels = LARGE if "inionsKilled" in feat else SMALL
        s["bin"] = pd.cut(s[feat] - s[feat + "_opp"], edges, labels=labels)
        t = s.groupby("bin", observed=True).agg(games=("win", "size"), win_rate=("win", "mean"),
                                                 MCI_mean=("MCI", "mean")).reset_index()
        t.insert(0, "feature", feat)
        t.insert(0, "role", role)
        rows.append(t)
        print(t.round(3).to_string(index=False))
    out = pretty(pd.concat(rows))
    out.to_csv(os.path.join(args.results_dir, "conclusion_check.csv"), index=False,
               encoding="utf-8-sig")  # 엑셀에서 한글이 깨지지 않도록
    print("완료")


if __name__ == "__main__":
    main()
