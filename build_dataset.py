"""
4단계: 라인별 학습용 표(X + Y) 만들기

입력 : extract_tci_csv.py 결과 (games / player_minute .csv.gz)
출력 : out_dir/lane_TOP.csv.gz, lane_JUNGLE.csv.gz, lane_MIDDLE.csv.gz, lane_BOTTOM.csv.gz
       out_dir/norm_stats.csv   (분별 평균·표준편차: 보고서/재현용)
       out_dir/split_games.csv  (게임별 train/test, fold 배정)

설계 (3단계 결정 사항)
  - 정제: 솔로랭크(420)만, 2025 시즌(15.x)만, 40분 이하, 분석 구간 5~40분, 종료 프레임 제외
  - 행 단위: (게임, 분, 라인) / 값: 블루 - 레드 격차
  - Y = z(골드 격차) + z(XP 격차)   (z는 분별 정규화, BOTTOM은 ADC+SPT 듀오 합산)
  - X : 라인별 목록 (LANE_FEATURES), Y에 쓰인 골드·XP·레벨은 제외
  - 로밍/갱킹 : 14분까지의 값에서 고정 (이후 한타 관여 배제)
  - 분할 : 게임 단위 train 80% / test 20%, train 안에서 게임 단위 5-fold
  - 정규화 통계(평균·표준편차)는 train 게임으로만 계산 → test 정보 누수 방지

사용 예
  python build_dataset.py --csv_dir ./csv_v2 --out_dir ./dataset
  python build_dataset.py --csv_dir ./csv_v2 --out_dir ./dataset_s --suffix _sample2000   # 표본으로 시험
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
MIN_MINUTE, MAX_MINUTE = 5, 40
MAX_DURATION_S = 2400
EARLY_CUT = 14  # 로밍/갱킹 고정 시점(분)

LANE_FEATURES = {
    "TOP": ["minionsKilled", "kills", "deaths", "assists", "soloKills", "plates",
            "dmgToChamp", "dmgTaken", "roamKA_e14"],
    "JUNGLE": ["jungleMinionsKilled", "kills", "deaths", "assists", "killPart", "objKA",
               "dmgToChamp", "wardsPlaced", "gankKA_e14"],
    "MIDDLE": ["minionsKilled", "kills", "deaths", "assists", "soloKills", "killPart", "objKA",
               "dmgToChamp", "dmgTaken", "ccEnemy_ms", "roamKA_e14"],
    "BOTTOM": ["adc_minionsKilled", "adc_kills", "adc_deaths", "adc_dmgToChamp",
               "spt_assists", "spt_deaths", "spt_killPart", "spt_wardsPlaced",
               "spt_wardsKilled", "spt_ccEnemy_ms", "spt_roamKA_e14"],
}

USECOLS = ["matchId", "minute", "participantId", "teamId", "teamPosition",
           "totalGold", "xp", "minionsKilled", "jungleMinionsKilled", "dmgToChamp", "dmgTaken",
           "ccEnemy_ms", "kills", "deaths", "assists", "soloKills", "teamKills", "objKA",
           "plates", "wardsPlaced", "wardsKilled",
           "ka_TOP", "ka_MID", "ka_BOT", "ka_JUNGLE"]

OWN_ZONE = {"TOP": "ka_TOP", "MIDDLE": "ka_MID", "BOTTOM": "ka_BOT", "UTILITY": "ka_BOT"}
ZONE_COLS = ["ka_TOP", "ka_MID", "ka_BOT", "ka_JUNGLE"]  # BASE는 로밍/갱킹에서 제외


def load(csv_dir, suffix):
    games = pd.read_csv(os.path.join(csv_dir, f"games{suffix}.csv.gz"))
    keep = games[(games.queueId == 420)
                 & (~games.gameVersion.astype(str).str.startswith("14"))
                 & (games.gameDuration_s <= MAX_DURATION_S)].copy()
    print(f"게임 정제: {len(games)} -> {len(keep)}")
    keep_ids = set(keep.matchId)

    # 큰 파일이라 나눠 읽으며 필요한 행만 남김 (메모리 절약)
    parts = []
    for ch in pd.read_csv(os.path.join(csv_dir, f"player_minute{suffix}.csv.gz"),
                          usecols=USECOLS, chunksize=1_000_000):
        ch = ch[ch.matchId.isin(keep_ids) & (ch.minute <= MAX_MINUTE + 1)]
        num = ch.select_dtypes("number").columns
        ch[num] = ch[num].astype("float32")
        parts.append(ch)
    pm = pd.concat(parts, ignore_index=True)
    return keep, pm


def add_player_features(pm):
    # 종료 프레임(게임 마지막 분)은 이벤트 몰림이 있어 제외
    last = pm.groupby("matchId").minute.transform("max")
    pm = pm[pm.minute < last].copy()

    pm["killPart"] = ((pm.kills + pm.assists) / pm.teamKills.replace(0, np.nan)).fillna(0)
    pm["gankKA"] = pm.ka_TOP + pm.ka_MID + pm.ka_BOT
    pm["roamKA"] = np.nan
    for pos, own in OWN_ZONE.items():
        m = pm.teamPosition == pos
        pm.loc[m, "roamKA"] = pm.loc[m, ZONE_COLS].sum(axis=1) - pm.loc[m, own]

    # 14분 이후에는 14분 값으로 고정
    pm = pm.sort_values(["matchId", "participantId", "minute"])
    grp = [pm.matchId, pm.participantId]
    for c in ["roamKA", "gankKA"]:
        pm[c + "_e14"] = pm[c].where(pm.minute <= EARLY_CUT).groupby(grp).ffill()
    return pm[pm.minute.between(MIN_MINUTE, MAX_MINUTE)]


def lane_diff(pm, position, prefix=""):
    cols = ["totalGold", "xp", "minionsKilled", "jungleMinionsKilled", "kills", "deaths", "assists",
            "soloKills", "killPart", "objKA", "plates", "dmgToChamp", "dmgTaken", "ccEnemy_ms",
            "wardsPlaced", "wardsKilled", "roamKA_e14", "gankKA_e14"]
    s = pm[pm.teamPosition == position]
    b = s[s.teamId == 100].set_index(["matchId", "minute"])[cols]
    r = s[s.teamId == 200].set_index(["matchId", "minute"])[cols]
    d = b - r
    d.columns = [prefix + c for c in d.columns]
    return d


def make_split(keep):
    rng = np.random.default_rng(SEED)
    ids = keep.matchId.sample(frac=1, random_state=SEED).to_numpy()
    n_test = int(round(len(ids) * 0.2))
    split = pd.DataFrame({"matchId": ids})
    split["split"] = ["test"] * n_test + ["train"] * (len(ids) - n_test)
    split["fold"] = -1
    tr = split.split == "train"
    split.loc[tr, "fold"] = rng.permutation(np.arange(tr.sum()) % 5)
    return split.merge(keep[["matchId", "blueWin"]], on="matchId")


def minute_z(df, cols, train_mask, lane, stats_rows):
    """분별 z-score. 평균·표준편차는 train 행으로만 계산."""
    out = pd.DataFrame(index=df.index)
    minutes = df.index.get_level_values("minute")
    tr = df[train_mask]
    tr_min = tr.index.get_level_values("minute")
    for c in cols:
        mu = tr[c].groupby(tr_min).mean()
        sd = tr[c].groupby(tr_min).std().replace(0, 1).fillna(1)
        out[c] = (df[c].to_numpy() - mu.reindex(minutes).to_numpy()) / sd.reindex(minutes).to_numpy()
        for m in mu.index:
            stats_rows.append([lane, c, int(m), float(mu[m]), float(sd[m])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--suffix", default="", help="표본 파일이면 _sample2000")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keep, pm = load(args.csv_dir, args.suffix)
    pm = add_player_features(pm)
    split = make_split(keep)
    split.to_csv(os.path.join(args.out_dir, "split_games.csv"), index=False)
    print(f"분할: train {(split.split == 'train').sum()}게임 / test {(split.split == 'test').sum()}게임")

    stats_rows = []
    for lane, feats in LANE_FEATURES.items():
        if lane == "BOTTOM":
            d = lane_diff(pm, "BOTTOM", "adc_").join(lane_diff(pm, "UTILITY", "spt_"), how="inner")
            d["gold_diff"] = d.adc_totalGold + d.spt_totalGold
            d["xp_diff"] = d.adc_xp + d.spt_xp
        else:
            d = lane_diff(pm, lane)
            d["gold_diff"] = d.totalGold
            d["xp_diff"] = d.xp
        d = d.dropna(subset=feats + ["gold_diff", "xp_diff"])

        ids = d.index.get_level_values("matchId")
        sp = split.set_index("matchId").reindex(ids)
        train_mask = (sp.split == "train").to_numpy()

        zy = minute_z(d, ["gold_diff", "xp_diff"], train_mask, lane, stats_rows)
        zx = minute_z(d, feats, train_mask, lane, stats_rows)

        out = zx.add_prefix("z_")
        out["Y"] = zy.gold_diff + zy.xp_diff
        out["gold_diff"] = d.gold_diff
        out["xp_diff"] = d.xp_diff
        out = out.reset_index()
        out["split"] = sp.split.to_numpy()
        out["fold"] = sp.fold.to_numpy()
        out["blueWin"] = sp.blueWin.to_numpy()

        path = os.path.join(args.out_dir, f"lane_{lane}.csv.gz")
        out.to_csv(path, index=False, compression="gzip")
        print(f"  {lane}: {len(out):,}행, X {len(feats)}개 -> {path}")

    pd.DataFrame(stats_rows, columns=["lane", "variable", "minute", "mean", "std"]) \
        .to_csv(os.path.join(args.out_dir, "norm_stats.csv"), index=False)
    print("완료")


if __name__ == "__main__":
    main()
