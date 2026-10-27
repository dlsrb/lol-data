"""
타겟 변수(y) 선정 - 전체 데이터 버전

교수님 제공 evaluate_y.py를 확장한 코드이다.
  evaluate_y.py      : 게임 종료 시점, 팀 합계 골드(또는 피해량)의 우위로 승패를 맞히는 정확도
  evaluate_y_full.py : 같은 방식을 (1) 라인별 맞상대 격차로, (2) 종료 시점과 10·15·20분 고정 시점으로,
                       (3) 골드·XP·피해량과 그 조합으로, (4) 정제 조건을 적용한 전체 게임으로 확장

입력 : csv_v2/games.csv.gz, csv_v2/player_minute.csv.gz  (extract_tci_csv.py 결과, 전체 데이터)
출력 : results/y_selection.csv     라인 x 시점 x 후보별 승패 부호 일치율
       results/y_selection_ci.csv  15분 기준 후보 간 정확도 차이의 95% 신뢰구간(부트스트랩)
       results/filter_counts.csv   정제 조건별 제외 건수

사용 예
  python evaluate_y_full.py --csv_dir ./csv_v2 --results_dir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
LANE_POS = {"TOP": ["TOP"], "JUNGLE": ["JUNGLE"], "MIDDLE": ["MIDDLE"], "BOTTOM": ["BOTTOM", "UTILITY"]}
MINUTES = [10, 15, 20]


def filter_games(games):
    """정제 조건과 조건별 제외 건수."""
    c_queue = games.queueId != 420
    c_patch = games.gameVersion.astype(str).str.startswith("14")
    c_long = games.gameDuration_s > 2400
    counts = pd.DataFrame({
        "condition": ["전체(20분 이상)", "자유랭크 등 솔로랭크 외", "2024 시즌 패치(14.x)", "40분 초과",
                      "제외 합계(중복 제거)", "최종"],
        "games": [len(games), int(c_queue.sum()), int(c_patch.sum()), int(c_long.sum()),
                  int((c_queue | c_patch | c_long).sum()), int((~(c_queue | c_patch | c_long)).sum())],
    })
    return games[~(c_queue | c_patch | c_long)], counts


def load_minutes(csv_dir, keep_ids):
    use = ["matchId", "minute", "teamId", "teamPosition", "totalGold", "xp", "dmgToChamp"]
    parts = []
    for ch in pd.read_csv(os.path.join(csv_dir, "player_minute.csv.gz"), usecols=use, chunksize=1_000_000):
        ch = ch[ch.matchId.isin(keep_ids)]
        ch[["totalGold", "xp", "dmgToChamp"]] = ch[["totalGold", "xp", "dmgToChamp"]].astype("float32")
        parts.append(ch)
    pm = pd.concat(parts, ignore_index=True)
    pm["is_end"] = pm.minute == pm.groupby("matchId").minute.transform("max")
    return pm


def lane_diffs(pm, lane, when):
    """블루 - 레드 격차 (BOTTOM은 ADC+SPT 합산). when: 'end' 또는 분."""
    s = pm[pm.teamPosition.isin(LANE_POS[lane])]
    s = s[s.is_end] if when == "end" else s[(s.minute == when) & ~s.is_end]
    t = s.groupby(["matchId", "teamId"])[["totalGold", "xp", "dmgToChamp"]].sum().unstack("teamId")
    d = pd.DataFrame({k: t[(k, 100)] - t[(k, 200)] for k in ["totalGold", "xp", "dmgToChamp"]}).dropna()
    z = d / d.std()  # 같은 시점 안에서 표준편차로 나눔 (격차는 평균이 0 근처라 부호가 보존됨)
    return pd.DataFrame({"gold": d.totalGold, "xp": d.xp, "dmg": d.dmgToChamp,
                         "gold+xp": z.totalGold + z.xp, "gold+dmg": z.totalGold + z.dmgToChamp})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    games = pd.read_csv(os.path.join(args.csv_dir, "games.csv.gz"))
    keep, counts = filter_games(games)
    counts.to_csv(os.path.join(args.results_dir, "filter_counts.csv"), index=False)
    print(counts.to_string(index=False))

    pm = load_minutes(args.csv_dir, set(keep.matchId))
    win = keep.set_index("matchId").blueWin
    rng = np.random.default_rng(SEED)
    rows, ci_rows = [], []
    cands = ["gold", "xp", "dmg", "gold+xp", "gold+dmg"]

    for when in ["end"] + MINUTES:
        for lane in LANE_POS:
            d = lane_diffs(pm, lane, when)
            y = win.reindex(d.index).to_numpy() == 1
            hit = {c: (d[c].to_numpy() > 0) == y for c in cands}  # 게임별 정답 여부
            rows.append({"when": when, "lane": lane, "n_games": len(d),
                         **{c: hit[c].mean() for c in cands}})
            if when == 15:  # 15분: 골드 단독 대비 차이의 신뢰구간
                n = len(d)
                idx = rng.integers(0, n, size=(args.n_boot, n))
                for c in ["xp", "gold+xp", "gold+dmg"]:
                    diff = (hit[c].astype(float) - hit["gold"].astype(float))[idx].mean(axis=1)
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    ci_rows.append({"lane": lane, "compare": f"{c} - gold", "diff": hit[c].mean() - hit["gold"].mean(),
                                    "ci_low": lo, "ci_high": hi})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.results_dir, "y_selection.csv"), index=False)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(os.path.join(args.results_dir, "y_selection_ci.csv"), index=False)
    print("\n승패 부호 일치율")
    print(res.round(3).to_string(index=False))
    print("\n15분: 골드 단독 대비 정확도 차이 (95% 신뢰구간)")
    print(ci.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
