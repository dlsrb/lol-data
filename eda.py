"""
보완 1차 - 5번: 탐색적 데이터 분석(EDA)

목적: 보고서에 나오는 지표(골드+XP 격차, 피처, MCI)와 확률(승률, 부호 일치율)이
      데이터에서 어떻게 생겨나는지 그림으로 설명한다.

입력 : dataset/lane_*.csv.gz, results/tci_coefficients.csv, results/y_selection.csv
출력 : (eda1 그림은 쓰지 않음) 골드 격차의 표준편차 수치만 eda_summary.csv에 저장 → 표준 점수가 필요한 이유
       images/eda2_target_compare.png     타겟 후보 비교: 각 경우의 1위 후보와의 정확도 차이 (results/y_selection.csv 필요)
       images/eda3_feature_corr.png       라인별 피처와 Y의 상관 → 피처 선정 근거
       images/eda4_winprob_by_mci.png     MCI 구간별 승률 산점도 (역할별)
       results/eda_summary.csv            그림에 쓰인 수치

사용 예
  python eda.py --data_dir ./dataset --results_dir ./results --img_dir ./images
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
COLORS = {"TOP": "tab:red", "JUNGLE": "tab:green", "MIDDLE": "tab:blue", "BOTTOM": "tab:orange",
          "ADC": "tab:orange", "SPT": "tab:purple"}


from matplotlib import font_manager


def set_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if f in names:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"
ROLE_COLOR = {"TOP": "#2a78d6", "JUNGLE": "#eb6834", "MIDDLE": "#1baf7a", "ADC": "#eda100", "SPT": "#e87ba4",
              "BOTTOM": "#eda100"}
ROLE_KR = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "ADC": "원딜(ADC)", "SPT": "서포터(SPT)", "BOTTOM": "바텀(듀오)"}

def style(ax):
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#bdbdbd")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(50, color="#9a9a9a", lw=1, ls="--", zorder=1)

def binned(x, win, edges):
    b = pd.cut(x, edges)
    p = pd.DataFrame({"b": b, "w": win}).groupby("b", observed=True).w.agg(["mean", "size"])
    p = p[p["size"] >= 30]
    return np.array([iv.mid for iv in p.index]), p["mean"].to_numpy() * 100, p["size"].to_numpy()

def fig_mci(m, path):
    """m: columns role, MCI, win (player-game rows)."""
    set_korean_font()
    roles = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SPT"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True)
    edges = np.arange(-3, 3.01, 0.5)
    for ax, r in zip(axes, roles):
        g = m[m.role == r]
        ax.axvspan(-3, 0, color="#f3f3f3", zorder=0)
        x, y, n = binned(g.MCI, g.win, edges)
        ax.scatter(x, y, s=30 + 220 * n / n.max(), color=ROLE_COLOR[r], edgecolor="white", linewidth=1.5, zorder=3)
        style(ax); ax.set_xlim(-3, 3); ax.set_xticks([-2, -1, 0, 1, 2])
        ax.axvline(0, color="#bdbdbd", lw=1, zorder=1)
        hi = g[g.MCI >= 1].win.mean() * 100; lo = g[g.MCI < -1].win.mean() * 100
        ax.set_title(ROLE_KR[r], fontsize=13, color=INK, fontweight="bold", pad=10)
        ax.text(2.9, 6, f"MCI ≥ 1 이면\n승률 {hi:.0f}%", ha="right", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(-2.9, 94, f"MCI < −1 이면\n승률 {lo:.0f}%", ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(-1.5, 52, "열세", ha="center", va="bottom", fontsize=9, color=MUTED)
        ax.text(1.5, 44, "우세", ha="center", va="top", fontsize=9, color=MUTED)
        ax.set_xlabel("MCI (0 = 맞상대와 대등)", fontsize=10, color=MUTED)
    axes[0].set_ylabel("승률", fontsize=11, color=MUTED, rotation=0, labelpad=18)
    fig.suptitle("맞상대보다 잘할수록(MCI가 클수록) 승률이 올라간다", fontsize=15, color=INK, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.905, "점 하나 = MCI가 0.5 간격 구간에 속한 게임들의 승률, 점이 클수록 게임 수가 많음. 회색 배경 = 맞상대보다 열세", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=130); plt.close(fig)

def load(data_dir):
    out = {}
    for lane in LANES:
        out[lane] = pd.read_csv(os.path.join(data_dir, f"lane_{lane}.csv.gz"))
    return out


def eda1_spread(data, rows):
    """분 시점별 골드 격차의 표준편차 (그림 없이 수치만 저장: README의 표준 점수 설명용)."""
    for lane, d in data.items():
        sd = d.groupby("minute").gold_diff.std()
        for m in (5, 15, 25, 35):
            rows.append(["eda1", lane, f"gold_diff_std@{m}", float(sd.get(m, np.nan))])


CAND = [("gold+xp", "골드+XP"), ("gold", "골드"), ("xp", "XP"), ("gold+dmg", "골드+피해"), ("dmg", "가한 피해")]
TIME_SHADE = {"10": "#9ec5f4", "15": "#3987e5", "20": "#104281"}


def fig_target_compare(y, path):
    """타겟 후보 비교: 각 경우(라인 x 시점)에서 1위 후보와 얼마나 차이 나는가 (0 = 그 경우 1위)."""
    cols = [c for c, _ in CAND]
    y = y.copy(); y["when"] = y["when"].astype(str)
    fixed, end = y[y["when"] != "end"].copy(), y[y["when"] == "end"].copy()
    gap_f = (fixed[cols].to_numpy() - fixed[cols].max(axis=1).to_numpy()[:, None]) * 100
    gap_e = (end[cols].to_numpy() - end[cols].max(axis=1).to_numpy()[:, None]) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(0)
    for k, (c, name) in enumerate(CAND):
        yk = len(CAND) - 1 - k
        hl = c == "gold+xp"
        if hl:
            for ax in (a1, a2):
                ax.axhspan(yk - 0.42, yk + 0.42, color="#eaf2fc", zorder=0)
        for i, w in enumerate(fixed["when"]):
            a1.scatter(gap_f[i, k], yk + rng.uniform(-0.18, 0.18), s=55, color=TIME_SHADE[w], edgecolor="white",
                       linewidth=1, zorder=3)
        a2.scatter(gap_e[:, k], yk + rng.uniform(-0.15, 0.15, len(gap_e)), s=55, color="#6b6b6b", edgecolor="white",
                   linewidth=1, zorder=3)
        n1 = int((gap_f[:, k] == 0).sum())
        a1.text(0.6, yk, f"평균 {fixed[c].mean()*100:.1f}%  ·  1위 {n1}회  ·  1위와 최대 {-gap_f[:, k].min():.1f}%p 차이",
                va="center", fontsize=9.5, color="#2b2b2b", fontweight="bold" if hl else "normal")
        a2.text(0.6, yk, f"평균 {end[c].mean()*100:.1f}%", va="center", fontsize=9.5, color="#2b2b2b",
                fontweight="bold" if hl else "normal")
    for ax, title in ((a1, "고정 시점 10·15·20분 (4라인 × 3시점 = 12개 경우)  ← 타겟 선택 기준"),
                      (a2, "게임 종료 시점 (4라인)  ← 이긴 팀이 벌린 격차가 섞여 부풀려짐")):
        ax.set_yticks(range(len(CAND)))
        ax.set_yticklabels([n for _, n in CAND][::-1], fontsize=11)
        for t in ax.get_yticklabels():
            if t.get_text() == "골드+XP":
                t.set_fontweight("bold")
        ax.axvline(0, color="#9a9a9a", lw=1, ls="--", zorder=1)
        ax.set_title(title, fontsize=11, loc="left", color="#2b2b2b", pad=10)
        ax.set_xlabel("그 경우의 1위 후보와의 정확도 차이 (%p, 0 = 1위)", fontsize=10, color="#6b6b6b")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
    a1.set_xlim(-11.5, 10.5); a2.set_xlim(-28, 14)
    a1.set_xticks([-10, -7.5, -5, -2.5, 0]); a2.set_xticks([-25, -20, -15, -10, -5, 0])
    for w, col in TIME_SHADE.items():
        a1.scatter([], [], s=55, color=col, label=f"{w}분")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, ncol=3, title="점 색 = 시점", title_fontsize=9)
    fig.suptitle("골드+XP는 고정 시점 12개 경우 모두에서 1위와 0.4%p 이내 — XP는 종료 시점에서만 1위",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", color="#2b2b2b")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=130); plt.close(fig)


def eda2_target(data, y_sel, path, rows, minutes=(10, 15, 20)):
    """타겟 후보 비교 그림(results/y_selection.csv 사용) + 골드+XP 격차의 승패 부호 일치율 수치."""
    for m in minutes:
        for lane, d in data.items():
            s = d[d.minute == m]
            rows.append(["eda2", lane, f"sign_acc@{m}", float(((s.Y > 0) == (s.blueWin == 1)).mean())])
    fig_target_compare(y_sel, path)


def eda3_corr(data, path, rows):
    """라인별 피처(격차, z-score)와 Y의 상관."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, lane in zip(axes, LANES):
        d = data[lane]
        feats = [c for c in d.columns if c.startswith("z_")]
        c = d[feats + ["Y"]].corr()["Y"].drop("Y").sort_values()
        ax.barh([f[2:] for f in c.index], c.values,
                color=["tab:red" if v < 0 else "tab:blue" for v in c.values])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(lane)
        for f, v in c.items():
            rows.append(["eda3", lane, f"corr_Y:{f[2:]}", float(v)])
    axes[0].set_xlabel("correlation with Y")
    fig.suptitle("Correlation between each feature and the target (all minutes, all games)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def eda4_mci(data, coef, path, rows):
    """게임 단위 MCI에 따른 승률 (역할별 산점도). BOTTOM은 ADC/SPT 항 분리."""
    parts = {"TOP": ("TOP", None), "JUNGLE": ("JUNGLE", None), "MIDDLE": ("MIDDLE", None),
             "ADC": ("BOTTOM", "adc_"), "SPT": ("BOTTOM", "spt_")}
    frames = []
    for role, (lane, pre) in parts.items():
        d = data[lane]
        w = coef[coef.lane == lane].set_index("feature").coef
        feats = [f for f in w.index if pre is None or f.startswith(pre)]
        d = d.assign(tci=d[["z_" + f for f in feats]].to_numpy() @ w[feats].to_numpy())
        g = d.groupby("matchId").agg(score=("tci", "mean"), win=("blueWin", "first"))
        # 블루 플레이어(점수 그대로)와 레드 플레이어(부호 반전, 승패 반전)를 모두 포함
        g = pd.concat([g, pd.DataFrame({"score": -g.score, "win": 1 - g.win})], ignore_index=True)
        g["MCI"] = g.score / g.score.std()
        g["role"] = role
        frames.append(g[["role", "MCI", "win"]])
        rows.append(["eda4", role, "P(win|MCI>=1)", float(g[g.MCI >= 1].win.mean())])
    fig_mci(pd.concat(frames, ignore_index=True), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    data = load(args.data_dir)
    coef = pd.read_csv(os.path.join(args.results_dir, "tci_coefficients.csv"))
    rows = []
    set_korean_font()
    eda1_spread(data, rows)
    y_sel = pd.read_csv(os.path.join(args.results_dir, "y_selection.csv"))  # evaluate_y_full.py 결과
    eda2_target(data, y_sel, os.path.join(args.img_dir, "eda2_target_compare.png"), rows)
    eda3_corr(data, os.path.join(args.img_dir, "eda3_feature_corr.png"), rows)
    eda4_mci(data, coef, os.path.join(args.img_dir, "eda4_winprob_by_mci.png"), rows)
    pd.DataFrame(rows, columns=["figure", "group", "metric", "value"]) \
        .to_csv(os.path.join(args.results_dir, "eda_summary.csv"), index=False)
    print("완료")


if __name__ == "__main__":
    main()
