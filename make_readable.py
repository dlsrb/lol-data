"""
결과 CSV를 사람이 읽기 쉬운 표로 바꿔 results_readable/ 에 저장한다.

- results/ 의 원본 CSV는 다른 코드가 읽어 쓰므로 그대로 둔다.
- results_readable/ 의 파일은 보고서 작성·발표용: 한글 열 이름, 한글 피처 이름, 퍼센트, 소수점 정리.
- 엑셀에서 한글이 깨지지 않도록 utf-8-sig로 저장한다.

사용 예
  python make_readable.py --results_dir ./results --out_dir ./results_readable
"""
import argparse
import os

import pandas as pd

FEAT = {"minionsKilled": "CS", "jungleMinionsKilled": "정글 CS", "kills": "킬", "deaths": "데스",
        "assists": "어시스트", "soloKills": "솔로킬", "plates": "포탑 방패", "dmgToChamp": "가한 피해",
        "dmgTaken": "받은 피해", "killPart": "킬 관여율", "objKA": "오브젝트 관여", "wardsPlaced": "와드 설치",
        "wardsKilled": "와드 제거", "ccEnemy_ms": "CC 시간", "roamKA": "로밍", "gankKA": "갱킹"}
MODEL = {"Baseline_CS": "기준선(CS만)", "Linear": "선형회귀", "Ridge": "릿지 회귀(최종)", "Lasso": "라쏘 회귀",
         "DecisionTree": "결정 나무", "RandomForest": "랜덤 포레스트", "XGBoost": "XGBoost", "LightGBM": "LightGBM"}
LANE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "ADC", "SPT"]


def feat_kr(f):
    f = f[2:] if f.startswith("z_") else f
    for pre, tag in (("adc_", "[ADC] "), ("spt_", "[SPT] ")):
        if f.startswith(pre):
            return tag + FEAT.get(f[len(pre):], f[len(pre):])
    return FEAT.get(f, f)


def lane_sort(df, col="라인"):
    return df.assign(_o=df[col].map({l: i for i, l in enumerate(LANE_ORDER)})).sort_values("_o", kind="stable").drop(columns="_o")


def model_performance(r):
    m = pd.read_csv(os.path.join(r, "metrics.csv"))
    return pd.DataFrame({
        "라인": m.lane, "모델": m.model.map(MODEL),
        "R²(설명력)": m.R2.round(3), "RMSE": m.RMSE.round(3), "MSE": (m.RMSE ** 2).round(3), "MAE": m.MAE.round(3),
        "골드 방향 일치율(%)": (m.SignAcc_gold * 100).round(1)})


def tci_weights(r):
    c = pd.read_csv(os.path.join(r, "tci_coefficients.csv"))
    return pd.DataFrame({
        "라인": c.lane, "피처": c.feature.map(feat_kr), "가중치": c.coef.round(3),
        "다시 학습 시 범위(하한)": c.ci_low.round(3), "다시 학습 시 범위(상한)": c.ci_high.round(3),
        "타겟과의 상관": c.corr_with_Y.round(2),
        "상관과 방향이 반대": c.sign_flip.map({True: "예", False: "아니오"})})


def feature_importance(r):
    f = pd.read_csv(os.path.join(r, "feature_importance.csv"))
    f["피처"] = f.feature.map(feat_kr)
    w = f.pivot_table(index=["lane", "피처"], columns="model", values="value", sort=False).reset_index()
    w = w.rename(columns={"lane": "라인", **{k: MODEL[k] + (" 계수" if k in ("Linear", "Ridge", "Lasso") else " 중요도")
                                            for k in MODEL if k in w.columns}})
    return lane_sort(w).round(3)


def model_compare(r):
    c = pd.read_csv(os.path.join(r, "compare_summary.csv"))
    rows = []
    for _, x in c.iterrows():
        if x.scope == "all":
            rows.append([x.lane, MODEL.get(x.model, x.model), "평가 데이터 전체", round(x.R2_or_low, 3), round(x.RMSE_or_high, 3), ""])
        elif x.scope.startswith("min"):
            rows.append([x.lane, MODEL.get(x.model, x.model), f"{x.scope[3:]}분 시점", "", round(x.RMSE_or_high, 3), ""])
        else:
            rows.append([x.lane, "XGBoost − 릿지", "R² 차이", "", "",
                         f"500번 다시 뽑아 계산한 범위 {x.R2_or_low * 100:.1f}~{x.RMSE_or_high * 100:.1f}%p"])
    return lane_sort(pd.DataFrame(rows, columns=["라인", "모델", "구간", "R²(설명력)", "RMSE", "비고"]))


def eda_summary(r):
    e = pd.read_csv(os.path.join(r, "eda_summary.csv"))
    out = []
    for _, x in e.iterrows():
        m = x.metric
        if x.figure == "eda1":
            out.append(["Fig. 1", x.group, f"{m.split('@')[1]}분 골드 격차의 표준편차(골드)", round(x.value)])
        elif x.figure == "eda2":
            out.append(["Fig. 2", x.group, f"{m.split('@')[1]}분 골드+XP 격차로 승패를 맞힌 비율(%)", round(x.value * 100, 1)])
        elif x.figure == "eda3":
            out.append(["Fig. 3", x.group, f"타겟과의 상관: {feat_kr(m.split(':')[1])}", round(x.value, 2)])
        else:
            out.append(["Fig. 7", x.group, "MCI ≥ 1인 게임의 승률(%)", round(x.value * 100, 1)])
    return pd.DataFrame(out, columns=["그림", "라인/역할", "항목", "값"])


def target_selection(r):
    y = pd.read_csv(os.path.join(r, "y_selection.csv"))
    when = {"end": "게임 종료"}
    return pd.DataFrame({
        "시점": y["when"].astype(str).map(lambda w: when.get(w, f"{w}분")), "라인": y.lane, "게임 수": y.n_games,
        "골드(%)": (y.gold * 100).round(1), "XP(%)": (y.xp * 100).round(1), "가한 피해(%)": (y.dmg * 100).round(1),
        "골드+XP(%)": (y["gold+xp"] * 100).round(1), "골드+피해(%)": (y["gold+dmg"] * 100).round(1)})


def target_compare(r):
    c = pd.read_csv(os.path.join(r, "y_selection_ci.csv"))
    name = {"xp - gold": "XP − 골드", "gold+xp - gold": "골드+XP − 골드", "gold+dmg - gold": "골드+피해 − 골드"}
    judge = c.apply(lambda x: "높음" if x.ci_low > 0 else ("낮음" if x.ci_high < 0 else "차이 없음"), axis=1)
    return pd.DataFrame({"라인": c["lane"], "비교 (15분 시점)": c["compare"].map(name),
                         "정확도 차이(%p)": (c["diff"] * 100).round(2), "판정 (1,000번 반복 계산)": judge})


def simple(r):
    ev = pd.read_csv(os.path.join(r, "extract_validation.csv"))
    ev_name = {"match rate: totalGold vs goldEarned": "골드: 타임라인 마지막 값 = 매치 기록 (일치율 %)",
               "match rate: kills vs kills_end": "킬: 타임라인 마지막 값 = 매치 기록 (일치율 %)",
               "match rate: deaths vs deaths_end": "데스: 타임라인 마지막 값 = 매치 기록 (일치율 %)",
               "match rate: assists vs assists_end": "어시스트: 타임라인 마지막 값 = 매치 기록 (일치율 %)",
               "match rate: wardsPlaced vs wardsPlaced_end": "와드 설치: 타임라인 마지막 값 = 매치 기록 (일치율 %)",
               "corr: ccEnemy_ms vs totalTimeCCDealt": "CC 시간: 타임라인(ms)과 매치 기록(초)의 상관",
               "median ratio: ccEnemy_ms / totalTimeCCDealt": "CC 시간: 타임라인 ÷ 매치 기록 (중앙값, 약 1,000배 = 단위 차이)",
               "players": "검증한 플레이어-게임 기록 수"}
    ev_val = [round(v * 100, 1) if k.startswith("match") else (round(v, 4) if k.startswith("corr") else round(v, 1))
              for k, v in zip(ev.metric, ev.value)]
    extract = pd.DataFrame({"항목": ev.metric.map(ev_name), "값": ev_val})

    fs = pd.read_csv(os.path.join(r, "feature_screen.csv"))
    fs_name = {"corr(soloKills, soloDeaths)": "솔로킬 격차와 솔로데스 격차의 상관 (−1이면 같은 정보)",
               "corr(objPart, Y)": "오브젝트 관여 비율과 타겟의 상관",
               "corr(objKA, Y)": "오브젝트 관여 횟수와 타겟의 상관",
               "corr(objPart, objKA)": "오브젝트 관여 비율과 횟수의 상관 (클수록 정보가 겹침)"}
    screen = pd.DataFrame({"라인": fs.lane, "항목": fs.metric.map(fs_name), "값": fs.value.round(2)})

    z = pd.read_csv(os.path.join(r, "zone_validation.csv"))
    zone = lane_sort(pd.DataFrame({"라인": z.lane.replace({"ALL": "전체"}), "10분 전 맞상대 간 킬 수": z.kills,
                                   "자기 라인 구역으로 분류된 킬 수": z.correct, "올바르게 분류된 비율(%)": (z.rate * 100).round(1)}))

    fc = pd.read_csv(os.path.join(r, "filter_counts.csv"), encoding="utf-8-sig")
    filt = pd.DataFrame({"정제 단계": fc.condition, "게임 수": fc.games})
    return extract, screen, zone, filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--out_dir", default="./results_readable")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    extract, screen, zone, filt = simple(a.results_dir)
    tables = {
        "01_데이터_정제_건수.csv": filt,
        "02_추출_검증.csv": extract,
        "03_타겟_후보별_승패_정확도.csv": target_selection(a.results_dir),
        "04_타겟_후보_15분_차이_판정.csv": target_compare(a.results_dir),
        "05_제외한_피처_근거.csv": screen,
        "06_로밍_갱킹_구역_검증.csv": zone,
        "07_모델별_성능.csv": model_performance(a.results_dir),
        "08_릿지_XGBoost_비교.csv": model_compare(a.results_dir),
        "09_TCI_가중치.csv": tci_weights(a.results_dir),
        "10_모델별_피처_가중치와_중요도.csv": feature_importance(a.results_dir),
        "11_그림에_쓰인_수치.csv": eda_summary(a.results_dir),
    }
    ft = os.path.join(a.results_dir, "feature_threshold.csv")
    if os.path.exists(ft):
        f = pd.read_csv(ft, encoding="utf-8-sig")
        tables["13_상관_0.5_기준_피처_실험.csv"] = pd.DataFrame({
            "라인": f.lane, "설정": f.setting, "피처 수": f.n_features, "R²(설명력)": f.R2.round(3),
            "RMSE": f.RMSE.round(3), "골드 방향 일치율(%)": (f.SignAcc_gold * 100).round(1),
            "빠진 피처": f.dropped.fillna("").map(lambda x: ", ".join(feat_kr(t.strip()) for t in x.split(",")) if x else "")})
    for i, (role, kr) in enumerate((("middle", "미드"), ("jungle", "정글")), start=14):
        sc = os.path.join(a.results_dir, f"summoner_compare_{role}.csv")
        if os.path.exists(sc):
            tables[f"{i}_{kr}_챌린저_두_명_실력_지표.csv"] = pd.read_csv(sc, encoding="utf-8-sig")
    cc = os.path.join(a.results_dir, "conclusion_check.csv")
    if os.path.exists(cc):
        tables["12_결론_라인별_핵심_플레이.csv"] = pd.read_csv(cc, encoding="utf-8-sig")
    for name, df in tables.items():
        df.to_csv(os.path.join(a.out_dir, name), index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)}행")
    print("완료:", a.out_dir)


if __name__ == "__main__":
    main()
