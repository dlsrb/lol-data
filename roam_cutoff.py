"""
보완 1차 - 3번: 로밍·갱킹의 14분 기준 근거

질문: 라인 밖 킬 관여를 "로밍"으로 볼 수 있는 시간은 언제까지인가?
방법: 각 분(1분 구간)에 새로 발생한 킬 관여(킬+어시스트) 중
      - 라이너(TOP/MIDDLE/ADC/SPT): 자기 라인 구역 밖에서 발생한 비율
      - 정글: 라인 구역(탑·미드·봇)에서 발생한 비율(갱킹 비율)
      을 계산해 시간에 따른 변화를 본다.
      라이너의 라인 밖 비율이 급격히 높아지는 시점 = 라인전이 끝나고 합류·한타가 시작되는 시점

입력 : csv_v2/games.csv.gz, csv_v2/player_minute.csv.gz (ka_* 열 필요)
출력 : images/roam_by_minute.png, results/roam_by_minute.csv

사용 예
  python roam_cutoff.py --csv_dir ./csv_v2 --results_dir ./results --img_dir ./images
  python roam_cutoff.py --csv_dir ./csv_v2 --results_dir ./results --img_dir ./images --suffix _sample2000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Z = ["ka_TOP", "ka_MID", "ka_BOT", "ka_JUNGLE", "ka_BASE"]
OWN = {"TOP": "ka_TOP", "MIDDLE": "ka_MID", "BOTTOM": "ka_BOT", "UTILITY": "ka_BOT"}
NAME = {"TOP": "TOP", "MIDDLE": "MIDDLE", "BOTTOM": "ADC", "UTILITY": "SPT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max_minute", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    g = pd.read_csv(os.path.join(args.csv_dir, f"games{args.suffix}.csv.gz"))
    keep = set(g[(g.queueId == 420) & (g.gameDuration_s <= 2400)
                 & (~g.gameVersion.astype(str).str.startswith("14"))].matchId)
    use = ["matchId", "participantId", "minute", "teamPosition"] + Z
    parts = [c[c.matchId.isin(keep) & (c.minute <= args.max_minute)]
             for c in pd.read_csv(os.path.join(args.csv_dir, f"player_minute{args.suffix}.csv.gz"),
                                  usecols=use, chunksize=1_000_000)]
    pm = pd.concat(parts).sort_values(["matchId", "participantId", "minute"])

    # 누적값 → 1분 구간 증가분
    inc = pm.groupby(["matchId", "participantId"])[Z].diff().fillna(pm[Z])
    inc["minute"], inc["pos"] = pm.minute.to_numpy(), pm.teamPosition.to_numpy()

    rows = []
    for m in range(3, args.max_minute + 1):  # 1~2분은 킬이 거의 없어 제외
        s = inc[inc.minute == m]
        r = {"minute": m}
        for pos, own in OWN.items():
            t = s[s.pos == pos]
            tot = t[Z].to_numpy().sum()
            r[NAME[pos] + "_outside_lane"] = (tot - t[own].sum()) / tot if tot else np.nan
        t = s[s.pos == "JUNGLE"]
        tot = t[Z].to_numpy().sum()
        r["JUNGLE_in_lane"] = t[["ka_TOP", "ka_MID", "ka_BOT"]].to_numpy().sum() / tot if tot else np.nan
        rows.append(r)
    res = pd.DataFrame(rows).set_index("minute")
    res.round(3).to_csv(os.path.join(args.results_dir, "roam_by_minute.csv"))
    print(res.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, col in zip(res.columns, ["tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:green"]):
        ax.plot(res.index, res[c], label=c.replace("_", " "), color=col, ls="--" if "JUNGLE" in c else "-")
    ax.axvline(14, color="black", lw=1)
    ax.text(14.3, 0.08, "14 min: turret plates fall", fontsize=9)
    ax.set_xlabel("minute (takedowns that occurred during that minute)")
    ax.set_ylabel("share of takedowns")
    ax.set_ylim(0, 1)
    ax.set_title("Where do takedowns happen? (laning phase vs. grouping phase)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.img_dir, "roam_by_minute.png"), dpi=120)


if __name__ == "__main__":
    main()
