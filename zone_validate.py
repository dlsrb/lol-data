"""
맵 구역(map_zone) 정의 검증 - 전체 데이터 버전

교수님 제공 data_process_sample.py와 같은 방식(match/timeline JSON을 읽고, teamPosition으로 라인을 구분하고,
frame["events"]의 CHAMPION_KILL을 순회)으로 작성했다.

검증 방법
  10분 이전에 같은 포지션의 맞상대끼리 발생한 킬(TOP-TOP, MIDDLE-MIDDLE, BOTTOM/UTILITY-BOTTOM/UTILITY)은
  대부분 해당 라인에서 일어난다. 이 킬의 좌표가 map_zone으로 해당 라인 구역에 분류되는 비율을 구한다.

입력 : match_data 폴더의 match_*.json, timeline_*.json (전체)
출력 : results/zone_validation.csv, results/zone_check.png (구역별 킬 위치, 1만 건 무작위 표시)

사용 예
  python zone_validate.py --input_dir ./match_data --results_dir ./results --img_dir ./images
"""
import argparse
import collections
import glob
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extract_tci_csv import map_zone  # 추출 코드와 같은 구역 정의를 사용

LANE_ZONE = {"TOP": "TOP", "MIDDLE": "MID", "BOTTOM": "BOT", "UTILITY": "BOT"}
EARLY_MS = 10 * 60 * 1000


def get_game_info(match):
    """정제 조건 통과 여부와 참가자별 (팀, 포지션)."""
    info = match["info"]
    ok = (info.get("queueId") == 420 and not str(info.get("gameVersion", "")).startswith("14")
          and 1200 <= info.get("gameDuration", 0) <= 2400)
    pos = {p["participantId"]: (p["teamId"], p["teamPosition"]) for p in info["participants"]}
    return ok, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    # 1) match: 게임별 정제 여부와 포지션만 저장
    games = {}
    for path in sorted(glob.glob(os.path.join(args.input_dir, "**", "*match_*.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            for m in json.load(f):
                games[m["metadata"]["matchId"]] = get_game_info(m)
    print(f"match 게임 수: {len(games)}")

    # 2) timeline: 10분 이전 맞상대 간 킬의 구역 분류
    total, correct = collections.Counter(), collections.Counter()
    sample_pts = []  # 그림용 (x, y, zone)
    random.seed(42)
    n_games = 0
    tl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*timeline_*.json"), recursive=True))
    for i, path in enumerate(tl_files, 1):
        with open(path, encoding="utf-8") as f:
            timelines = json.load(f)
        for tl in timelines:
            ok, pos = games.get(tl["metadata"]["matchId"], (False, None))
            if not ok:
                continue
            n_games += 1
            for frame in tl["info"]["frames"]:
                for event in frame["events"]:
                    if event["type"] != "CHAMPION_KILL" or "position" not in event:
                        continue
                    zone = map_zone(event["position"]["x"], event["position"]["y"])
                    if random.random() < 0.02:
                        sample_pts.append((event["position"]["x"], event["position"]["y"], zone))
                    k, v = event.get("killerId", 0), event.get("victimId", 0)
                    if event["timestamp"] >= EARLY_MS or k not in pos or v not in pos:
                        continue
                    (kt, kp), (vt, vp) = pos[k], pos[v]
                    same_lane = kp in LANE_ZONE and vp in LANE_ZONE and LANE_ZONE[kp] == LANE_ZONE[vp]
                    if kt != vt and same_lane:
                        lane = "BOTTOM" if kp in ("BOTTOM", "UTILITY") else kp
                        total[lane] += 1
                        correct[lane] += zone == LANE_ZONE[kp]
        if i % 50 == 0 or i == len(tl_files):
            print(f"  timeline {i}/{len(tl_files)} 파일, 게임 {n_games}개", flush=True)

    rows = [{"lane": l, "kills": total[l], "correct": correct[l], "rate": correct[l] / total[l]} for l in total]
    rows.append({"lane": "ALL", "kills": sum(total.values()), "correct": sum(correct.values()),
                 "rate": sum(correct.values()) / sum(total.values())})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.results_dir, "zone_validation.csv"), index=False)
    print(f"\n게임 {n_games}개, 10분 이전 맞상대 간 킬의 구역 분류 정확도")
    print(res.round(3).to_string(index=False))

    # 3) 그림: 구역별 킬 위치 (무작위 일부)
    pts = random.sample(sample_pts, min(10000, len(sample_pts)))
    col = {"TOP": "tab:red", "MID": "tab:green", "BOT": "tab:blue", "JUNGLE": "tab:gray", "BASE": "black"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for z, c in col.items():
        p = [q for q in pts if q[2] == z]
        ax.scatter([q[0] for q in p], [q[1] for q in p], s=2, c=c, label=z, alpha=0.6)
    ax.set_xlim(0, 14800)
    ax.set_ylim(0, 14800)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", markerscale=4)
    ax.set_title(f"Kill positions by map zone ({len(pts):,} random kills)")
    fig.savefig(os.path.join(args.img_dir, "zone_check.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
