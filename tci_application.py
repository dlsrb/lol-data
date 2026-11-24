"""
7단계: TCI 활용 - 매치 기여 지수(MCI)와 소환사 프로파일

입력 : dataset/lane_*.csv.gz           (분 단위 z-score X)
       results/tci_coefficients.csv     (최종 TCI = Ridge 계수)
       csv_v2/player_end.csv.gz         (게임별 참가자 puuid, 포지션, 승패)
       puuid_challengers.txt (선택)     (챌린저 표시용)
출력 : results/mci_games.csv.gz         (게임 x 플레이어 1행: MCI)
       results/summoner_profile.csv     (소환사 x 라인: 게임 수, MCI 평균·표준편차, 승률)
       results/mci_profile.png          (라인별 평균-기복 산점도, 논문 Fig.2 확장)

정의
  ΔTCI(t)  = Σ w_j · z_j(t)            (블루 - 레드 격차 기준, 절편 0)
  게임 점수 = 해당 게임의 ΔTCI 분 평균  → 블루 플레이어는 그대로, 레드 플레이어는 부호 반전
  MCI      = 게임 점수 / (전체 게임 점수의 표준편차)
             0 = 맞상대와 대등, +1 = 맞상대보다 1 표준편차만큼 우세
  BOTTOM   : 선형 모델이라 ADC 항(adc_*)과 SPT 항(spt_*)으로 나눌 수 있으므로
             ADC는 ADC 항 합계, SPT는 SPT 항 합계를 각자의 점수로 사용

참고: TCI는 학습이 끝난 고정된 식이므로, 활용 단계에서는 train/test 구분 없이 전체 게임에 적용한다.
      (성능 검증은 3.2절에서 test로만 수행함)

사용 예
  python tci_application.py --data_dir ./dataset --results_dir ./results --csv_dir ./csv_v2 --challengers puuid_challengers.txt
  python tci_application.py ... --challengers puuid_challengers.txt --challengers_only   # 프로파일을 챌린저로 한정
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
POS_OF = {"TOP": ["TOP"], "JUNGLE": ["JUNGLE"], "MIDDLE": ["MIDDLE"], "BOTTOM": ["BOTTOM", "UTILITY"]}


def game_scores(lane, data_dir, coef):
    """게임별 ΔTCI 분 평균 (블루 기준). BOTTOM은 ADC/SPT 항을 분리."""
    d = pd.read_csv(os.path.join(data_dir, f"lane_{lane}.csv.gz"))
    w = coef[coef.lane == lane].set_index("feature").coef
    if lane == "BOTTOM":
        parts = {"BOTTOM": [f for f in w.index if f.startswith("adc_")],
                 "UTILITY": [f for f in w.index if f.startswith("spt_")]}
    else:
        parts = {lane: list(w.index)}
    out = []
    for pos, feats in parts.items():
        tci = d[["z_" + f for f in feats]].to_numpy() @ w[feats].to_numpy()
        g = pd.DataFrame({"matchId": d.matchId, "tci": tci}).groupby("matchId").tci.mean()
        out.append(pd.DataFrame({"matchId": g.index, "teamPosition": pos, "blue_score": g.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def plot_profile(big, path, title):
    """산점도: x=평균(기여 수준), y=표준편차(기복), 크기=게임 수, 색=승률."""
    roles = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SPT"]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharex=True, sharey=True)
    for ax, role in zip(axes, roles):
        g = big[big.role == role]
        sc = ax.scatter(g.MCI_mean, g.MCI_std, s=g.games / 3, c=g.win_rate, cmap="viridis",
                        vmin=0.4, vmax=0.65, alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{role} (n={len(g)})")
        ax.set_xlabel("MCI mean (contribution)")
    axes[0].set_ylabel("MCI std (inconsistency)")
    fig.colorbar(sc, ax=axes, label="win rate", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--challengers", default="")
    ap.add_argument("--min_games", type=int, default=50, help="프로파일에 포함할 최소 게임 수 (논문: 50)")
    ap.add_argument("--challengers_only", action="store_true",
                    help="프로파일 분석·그림을 챌린저로 한정 (비챌린저는 챌린저와 함께한 게임만 있어 표본이 편향됨)")
    args = ap.parse_args()

    coef = pd.read_csv(os.path.join(args.results_dir, "tci_coefficients.csv"))
    end = pd.read_csv(os.path.join(args.csv_dir, "player_end.csv.gz"),
                      usecols=["matchId", "teamId", "teamPosition", "puuid", "win"])

    rows = []
    for lane in LANES:
        s = game_scores(lane, args.data_dir, coef)
        p = end[end.teamPosition.isin(POS_OF[lane])].merge(s, on=["matchId", "teamPosition"])
        p["score"] = np.where(p.teamId == 100, p.blue_score, -p.blue_score)  # 레드는 부호 반전
        p["lane"] = lane
        p["role"] = p.teamPosition.replace({"BOTTOM": "ADC", "UTILITY": "SPT"})
        rows.append(p)
    games = pd.concat(rows, ignore_index=True)

    # 역할별로 표준편차 1이 되도록 스케일 (라인 간 비교 가능하게)
    games["MCI"] = games.score / games.groupby("role").score.transform("std")
    # 새 게임(예: 내 경기)에 같은 기준을 적용할 수 있도록 역할별 나눗셈 기준을 저장 (my_mci.py에서 사용)
    games.groupby("role").score.std().rename("score_std").reset_index() \
        .to_csv(os.path.join(args.results_dir, "mci_scale.csv"), index=False)
    games = games[["matchId", "lane", "role", "teamId", "puuid", "win", "MCI"]]
    games.to_csv(os.path.join(args.results_dir, "mci_games.csv.gz"), index=False, compression="gzip")

    # 게임 단위 점검: MCI가 높을수록 이기는가
    print("게임 단위: MCI 구간별 승률")
    games["MCI_bin"] = pd.cut(games.MCI, [-np.inf, -1, 0, 1, np.inf], labels=["<-1", "-1~0", "0~1", ">1"])
    print(games.pivot_table(index="role", columns="MCI_bin", values="win", aggfunc="mean", observed=True)
          .round(3).to_string())

    # 소환사 프로파일
    prof = games.groupby(["role", "puuid"]).agg(games=("MCI", "size"), MCI_mean=("MCI", "mean"),
                                                MCI_std=("MCI", "std"), win_rate=("win", "mean")).reset_index()
    if args.challengers and os.path.exists(args.challengers):
        ch = set(open(args.challengers, encoding="utf-8").read().split())
        prof["challenger"] = prof.puuid.isin(ch)
    prof.to_csv(os.path.join(args.results_dir, "summoner_profile.csv"), index=False)

    big = prof[prof.games >= args.min_games]
    if "challenger" in big:
        print(f"\n{args.min_games}게임 이상: 챌린저 vs 그 외 (소환사 x 역할 기준)")
        print(big.groupby("challenger").agg(n=("puuid", "size"), MCI_mean=("MCI_mean", "mean"),
                                           win_rate=("win_rate", "mean")).round(3).to_string())
    if args.challengers_only:
        if "challenger" not in big:
            raise SystemExit("--challengers_only에는 --challengers 파일이 필요합니다.")
        big = big[big.challenger]
    print(f"\n{args.min_games}게임 이상 소환사 수 / MCI 평균과 승률의 상관")
    for role, g in big.groupby("role"):
        r = np.corrcoef(g.MCI_mean, g.win_rate)[0, 1] if len(g) > 2 else np.nan
        print(f"  {role:7s} {len(g):4d}명  r = {r:.3f}")

    name = "mci_profile_challengers.png" if args.challengers_only else "mci_profile.png"
    who = "challengers, " if args.challengers_only else ""
    plot_profile(big, os.path.join(args.results_dir, name),
                 f"Summoner MCI profile by role ({who}games >= {args.min_games})")
    print("\n완료:", args.results_dir)


if __name__ == "__main__":
    main()