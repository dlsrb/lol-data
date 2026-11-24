"""
맵 구역(map_zone) 정의 검증 - 전체 데이터 버전

교수님 제공 data_process_sample.py와 같은 방식(match/timeline JSON을 읽고, teamPosition으로 라인을 구분하고,
frame["events"]의 CHAMPION_KILL을 순회)으로 작성했다.

검증 방법
  10분 이전에 같은 포지션의 맞상대끼리 발생한 킬(TOP-TOP, MIDDLE-MIDDLE, BOTTOM/UTILITY-BOTTOM/UTILITY)은
  대부분 해당 라인에서 일어난다. 이 킬의 좌표가 map_zone으로 해당 라인 구역에 분류되는 비율을 구한다.

입력 : match_data 폴더의 match_*.json, timeline_*.json (전체)
출력 : results/zone_validation.csv, images/zone_check.png (구역별 킬 위치 1만 건 + 구역 경계)

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


from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import numpy as np


def set_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


from matplotlib.colors import ListedColormap

ZONE_COLOR = {"TOP": "tab:red", "MID": "tab:green", "BOT": "tab:blue", "JUNGLE": "tab:gray", "BASE": "black"}


def plot_zone_map(map_zone, pts, path, max_xy=14800, step=40):
    """원래 그림(구역별 색 점)에 구역 영역을 연하게 깔고 경계선을 그린다. 경계선은 구역 색, 범례는 그림 밖, 본진은 검정으로 통일하고 블루/레드를 글씨로 표시."""
    zones = list(ZONE_COLOR)
    xs = np.arange(0, max_xy + step, step)
    grid = np.array([[zones.index(map_zone(x, y)) for x in xs] for y in xs])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, origin="lower", extent=(0, max_xy, 0, max_xy), cmap=ListedColormap(list(ZONE_COLOR.values())),
              vmin=-0.5, vmax=len(zones) - 0.5, interpolation="nearest", alpha=0.08, zorder=0)
    # 구역 경계선: 각 구역의 색으로 그림 (정글 → 라인 → 본진 순서로 겹쳐 그려 라인·본진 경계가 위에 보이도록)
    for z in ["JUNGLE", "TOP", "MID", "BOT", "BASE"]:
        ax.contour(xs, xs, (grid == zones.index(z)).astype(float), levels=[0.5], colors=[ZONE_COLOR[z]],
                   linewidths=1.3, alpha=0.85, zorder=1)
    for z, c in ZONE_COLOR.items():
        p = [q for q in pts if q[2] == z]
        ax.scatter([q[0] for q in p], [q[1] for q in p], s=2, c=c, label=z, alpha=0.6, zorder=2)
    ax.text(2100, 2100, "BLUE\nBASE", ha="center", va="center", fontsize=10, fontweight="bold", color="black",
            zorder=3, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.text(12700, 12700, "RED\nBASE", ha="center", va="center", fontsize=10, fontweight="bold", color="black",
            zorder=3, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xlim(0, max_xy)
    ax.set_ylim(0, max_xy)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), markerscale=4, frameon=False, title="Kill zone")
    ax.set_title(f"Kill positions by map zone ({len(pts):,} random kills)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


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

    # 3) 그림: 구역을 색 영역으로 칠하고 실제 킬 위치(무작위 1만 건)를 겹쳐 그림
    pts = random.sample(sample_pts, min(10000, len(sample_pts)))
    plot_zone_map(map_zone, pts, os.path.join(args.img_dir, "zone_check.png"))


if __name__ == "__main__":
    main()
