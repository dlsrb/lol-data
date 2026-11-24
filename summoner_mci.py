"""
한 라인에서 소환사의 실력 지표 만들기: 승률이 같은 두 챌린저를 MCI로 비교 (--role로 라인 선택)

입력 : results/mci_games.csv.gz, puuid_challengers.txt
출력 : results/summoner_compare_<role>.csv  (두 소환사의 실력 지표)
       images/summoner_compare_<role>.png    (최근 50게임의 게임별 MCI)
선정 : 해당 라인 50게임 이상 챌린저 중 100게임 이상인 소환사끼리, 승률 차이 1%p 이내이면서 평균 MCI 차이가 가장 큰 두 명
       (소환사는 익명으로 '챌린저 A', '챌린저 B'로 표기)
사용 예
  python summoner_mci.py --role MIDDLE
  python summoner_mci.py --role JUNGLE
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def set_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


def profile(g, ref_means, role_kr):
    lose, win = g[g.win == 0], g[g.win == 1]
    return {"게임 수": len(g), "승률(%)": round(g.win.mean() * 100, 1), "평균 MCI": round(g.MCI.mean(), 2),
            "맞상대보다 우세한 게임 비율(%)": round((g.MCI > 0).mean() * 100, 1),
            "진 게임 중 우세했던 비율(%)": round((lose.MCI > 0).mean() * 100, 1),
            "이긴 게임 중 열세였던 비율(%)": round((win.MCI < 0).mean() * 100, 1),
            "MCI 표준편차(기복)": round(g.MCI.std(), 2),
            f"{role_kr} 50게임 이상 {len(ref_means)}명 중 평균 MCI 백분위": round((ref_means < g.MCI.mean()).mean() * 100, 1)}


ROLE_KR = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "ADC": "원딜", "SPT": "서포터"}


def rank_text(pct):
    v = 100 - pct if pct >= 50 else pct
    txt = f"{v:.1f}" if v < 1 else f"{v:.0f}"
    return f"상위 {txt}%" if pct >= 50 else f"하위 {txt}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--img_dir", default="./images")
    ap.add_argument("--challengers", default="puuid_challengers.txt")
    ap.add_argument("--role", default="MIDDLE")
    ap.add_argument("--recent", type=int, default=50)
    a = ap.parse_args()
    set_korean_font()

    m = pd.read_csv(os.path.join(a.results_dir, "mci_games.csv.gz"))
    m = m[m.role == a.role].copy()
    m["order"] = m.matchId.str.split("_").str[1].astype(np.int64)  # 매치 번호가 클수록 나중 게임
    ch = set(open(a.challengers, encoding="utf-8").read().split())
    p = m.groupby("puuid").agg(n=("MCI", "size"), mci=("MCI", "mean"), win=("win", "mean")).reset_index()
    big = p[p.n >= 50]
    ref_means = big.mci.to_numpy()
    c = big[big.puuid.isin(ch) & (big.n >= 100)].reset_index(drop=True)
    best = max(((abs(c.mci[i] - c.mci[j]), i, j) for i in range(len(c)) for j in range(i + 1, len(c))
                if abs(c.win[i] - c.win[j]) <= 0.01), key=lambda t: t[0])
    pair = c.loc[[best[1], best[2]]].sort_values("mci", ascending=False)
    labels = ["챌린저 A", "챌린저 B"]

    role_kr = ROLE_KR[a.role]
    table = pd.DataFrame({lab: profile(m[m.puuid == pu], ref_means, role_kr) for lab, pu in zip(labels, pair.puuid)})
    pct_key = f"{role_kr} 50게임 이상 {len(ref_means)}명 중 평균 MCI 백분위"
    table.index.name = "지표"
    table.to_csv(os.path.join(a.results_dir, f"summoner_compare_{a.role.lower()}.csv"), encoding="utf-8-sig")
    print(table.to_string())

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax, lab, pu in zip(axes, labels, pair.puuid):
        g = m[m.puuid == pu].sort_values("order").tail(a.recent).reset_index(drop=True)
        colors = np.where(g.win == 1, "#2a78d6", "#eb6834")
        ax.bar(np.arange(len(g)) + 1, g.MCI, color=colors, width=0.75, zorder=3)
        ax.axhline(0, color="#9a9a9a", lw=1)
        t = table[lab]
        ax.axhline(t["평균 MCI"], color="#2b2b2b", lw=1.2, ls="--", zorder=4)
        ax.text(len(g) + 0.8, t["평균 MCI"] + 0.12, f"평균 {t['평균 MCI']:+.2f}", va="bottom", fontsize=10, color="#2b2b2b")
        ax.set_title(f"{lab}   승률 {t['승률(%)']:.1f}%  |  평균 MCI {t['평균 MCI']:+.2f}  |  맞상대보다 우세한 게임 "
                     f"{t['맞상대보다 우세한 게임 비율(%)']:.0f}%  |  {role_kr} {len(ref_means)}명 중 {rank_text(t[pct_key])}",
                     loc="left", fontsize=11.5, color="#2b2b2b", fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
        ax.set_ylabel("MCI", rotation=0, labelpad=16, color="#6b6b6b")
    axes[1].set_xlabel(f"최근 {a.recent}게임 (왼쪽이 오래된 게임)", color="#6b6b6b")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#2a78d6", label="이긴 게임"), Patch(color="#eb6834", label="진 게임")],
                   loc="upper left", frameon=False, ncol=2, fontsize=10)
    fig.suptitle(f"승률은 같아도 맞상대 대비 기여(MCI)는 다르다 — {role_kr} 챌린저 두 명의 게임별 MCI",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", color="#2b2b2b")
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    fig.savefig(os.path.join(a.img_dir, f"summoner_compare_{a.role.lower()}.png"), dpi=130)


if __name__ == "__main__":
    main()
