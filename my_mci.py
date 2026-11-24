"""
내 경기(또는 아무 게임)의 MCI 계산

연구에서 학습한 TCI(릿지 가중치)와 정규화 기준(학습 데이터의 분별 평균·표준편차)을 그대로 써서,
새 게임의 소환사별 MCI를 계산한다. 모델을 다시 학습하지 않는다.

입력 : --match, --timeline   Riot API에서 받은 match / timeline JSON (게임 1개 또는 여러 게임의 리스트)
       results/tci_coefficients.csv, results/mci_scale.csv, dataset/norm_stats.csv  (연구 결과)
       results/mci_games.csv.gz (선택: 연구 데이터와 비교한 백분위 계산용)
출력 : 화면 표 + --out CSV (게임 x 플레이어: 역할, 챔피언, 승패, MCI, 같은 역할 연구 데이터 대비 백분위)

사용 예
  # 1) fetch_my_games.py로 내 경기 받기  →  my_match.json, my_timeline.json
  # 2) 내 경기만 보기 (Riot ID의 게임 이름으로 거르기)
  python my_mci.py --match my_match.json --timeline my_timeline.json --name 내게임이름
  # 연구 데이터 파일로 검증
  python my_mci.py --match match_0.json --timeline timeline_0.json

주의
  - 모델은 2025년 1~4월(패치 15.1~15.7) 한국 챌린저 솔로랭크로 학습했다. 다른 패치·티어의 게임에서는
    규칙과 플레이 양상이 달라 참고용으로만 해석해야 한다.
  - 5분 이전에 끝난 게임, 포지션이 비정상인 게임은 계산하지 않는다.
"""
import argparse
import collections
import csv
import io
import json
import os

import numpy as np
import pandas as pd

import build_dataset as bd
import extract_tci_csv as ex

LANES = {"TOP": ["TOP"], "JUNGLE": ["JUNGLE"], "MIDDLE": ["MIDDLE"], "BOTTOM": ["BOTTOM", "UTILITY"]}
ROLE = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "BOTTOM": "ADC", "UTILITY": "SPT"}


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [d] if isinstance(d, dict) else d


def extract(matches, timelines):
    """extract_tci_csv.process_game을 그대로 써서 분 단위 누적표를 만든다."""
    tl = {t["metadata"]["matchId"]: t for t in timelines}
    bufs = {k: io.StringIO() for k in ("g", "p", "e")}
    w = {k: csv.writer(b) for k, b in bufs.items()}
    w["g"].writerow(ex.GAME_COLS); w["p"].writerow(ex.PM_COLS); w["e"].writerow(ex.END_COLS)
    warn = collections.Counter()
    names = {}
    for m in matches:
        mid = m["metadata"]["matchId"]
        for p in m["info"]["participants"]:
            names[(mid, p["participantId"])] = (p.get("riotIdGameName") or p.get("summonerName") or "",
                                                p.get("championName"))
        if mid not in tl:
            print(f"[건너뜀] {mid}: timeline 없음"); continue
        ex.process_game(ex.slim_match(m), tl[mid], w["g"], w["p"], w["e"], warn)
    return [pd.read_csv(io.StringIO(bufs[k].getvalue())) for k in ("g", "p", "e")], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--name", default="", help="Riot ID 게임 이름(태그 제외). 지정하면 그 소환사만 출력")
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--data_dir", default="./dataset")
    ap.add_argument("--out", default="./results/my_mci.csv")
    a = ap.parse_args()

    (games, pm, end), names = extract(load(a.match), load(a.timeline))
    games = games[(games.positionsValid == 1) & (games.n_frames > 6)]
    pm = pm[pm.matchId.isin(games.matchId)]
    pm = bd.add_player_features(pm)

    coef = pd.read_csv(os.path.join(a.results_dir, "tci_coefficients.csv"))
    scale = pd.read_csv(os.path.join(a.results_dir, "mci_scale.csv")).set_index("role").score_std
    norm = pd.read_csv(os.path.join(a.data_dir, "norm_stats.csv"))

    rows = []
    for lane, feats in bd.LANE_FEATURES.items():
        if lane == "BOTTOM":
            d = bd.lane_diff(pm, "BOTTOM", "adc_").join(bd.lane_diff(pm, "UTILITY", "spt_"), how="inner")
        else:
            d = bd.lane_diff(pm, lane)
        d = d.dropna(subset=feats)
        minutes = d.index.get_level_values("minute")
        for f in feats:  # 연구의 학습 데이터 기준으로 분별 표준 점수
            st = norm[(norm.lane == lane) & (norm.variable == f)].set_index("minute")
            d["z_" + f] = (d[f].to_numpy() - st["mean"].reindex(minutes).to_numpy()) / st["std"].reindex(minutes).to_numpy()
        wts = coef[coef.lane == lane].set_index("feature").coef
        parts = ({"BOTTOM": [f for f in feats if f.startswith("adc_")], "UTILITY": [f for f in feats if f.startswith("spt_")]}
                 if lane == "BOTTOM" else {lane: feats})
        for pos, fs in parts.items():
            tci = d[["z_" + f for f in fs]].to_numpy() @ wts[fs].to_numpy()
            blue = pd.Series(tci, index=d.index.get_level_values("matchId")).groupby(level=0).mean()
            for mid, score in blue.items():
                for team, sign in ((100, 1), (200, -1)):
                    rows.append([mid, pos, team, sign * score / scale[ROLE[pos]]])
    mci = pd.DataFrame(rows, columns=["matchId", "teamPosition", "teamId", "MCI"])

    out = end.merge(mci, on=["matchId", "teamPosition", "teamId"])
    out["role"] = out.teamPosition.map(ROLE)
    out["name"] = [names.get((m, p), ("", ""))[0] for m, p in zip(out.matchId, out.participantId)]
    out["champion"] = [names.get((m, p), ("", ""))[1] for m, p in zip(out.matchId, out.participantId)]

    ref_path = os.path.join(a.results_dir, "mci_games.csv.gz")
    if os.path.exists(ref_path):  # 연구 데이터(같은 역할) 안에서 몇 % 위치인지
        ref = pd.read_csv(ref_path, usecols=["role", "MCI"])
        out["percentile"] = [round((ref.MCI[ref.role == r] < v).mean() * 100, 1) for r, v in zip(out.role, out.MCI)]

    if a.name:
        out = out[out.name.str.lower() == a.name.lower()]
    out = out.sort_values(["matchId", "teamId", "role"])
    cols = ["matchId", "name", "champion", "role", "win", "MCI"] + (["percentile"] if "percentile" in out else [])
    res = out[cols].assign(MCI=lambda d: d.MCI.round(2))
    res.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    if a.name and len(res):
        print(f"\n{a.name}: {len(res)}게임, 평균 MCI {res.MCI.mean():+.2f}, "
              f"맞상대보다 우세(MCI>0) {(res.MCI > 0).mean() * 100:.0f}%, 승률 {res.win.mean() * 100:.0f}%")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
