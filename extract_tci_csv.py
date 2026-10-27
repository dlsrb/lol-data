"""
LoL Match-V5 match/timeline JSON -> 분석용 CSV 추출

입력: match_*.json, timeline_*.json (각 파일은 게임 객체의 리스트)
출력 (gzip 압축 CSV, pandas.read_csv로 바로 읽힘):
  1) games.csv.gz        : 게임 1행 (메타 정보, 승패, 필터용 플래그)
  2) player_minute.csv.gz: (게임, 분, 참가자) 1행 — 누적값
  3) player_end.csv.gz   : (게임, 참가자) 1행 — match의 종료 시점 값 (검증용)

구역(map_zone) 정의
  킬 이벤트 좌표를 TOP/MID/BOT/JUNGLE(강가 포함)/BASE로 나눈 근사 기준.
  검증(30게임): 10분 이전 맞상대 간 킬의 88%(114/130)가 해당 라인 구역으로 분류됨.
  로밍(라이너) = 자기 라인 외 구역 관여, 갱킹(정글) = 라인 구역 관여 → 분석 단계에서 계산.

설계 원칙
  - 격차/정규화/Y 계산은 하지 않고 '원재료(누적값)'만 뽑는다.
    → Y 후보, 격차 방식, BOTTOM 듀오 합산 여부는 분석 단계에서 직접 결정.
  - 이벤트 누적값은 '이벤트 timestamp <= 프레임 timestamp' 기준으로 센다.
    (프레임 스냅샷 시점과 이벤트 집계 시점을 일치시키기 위함)

사용 예
  python extract_tci_csv.py --input_dir ./data --out_dir ./csv
  python extract_tci_csv.py --input_dir ./data --out_dir ./csv --match_glob "match_*.json" --timeline_glob "timeline_*.json"
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import time

def map_zone(x, y):
    """소환사의 협곡 좌표(약 0~14,800) -> 구역. 경계는 연구자가 정한 근사 기준."""
    if x is None or y is None:
        return "UNKNOWN"
    if (x < 4200 and y < 4200) or (x > 10600 and y > 10600):
        return "BASE"                       # 양 팀 본진(포탑 3·억제기 구역 포함)
    if (x < 2400 and y >= 4200) or (y > 12400 and x <= 10600) or (y - x > 8000):
        return "TOP"                        # 왼쪽 세로 + 위쪽 가로 띠 + 좌상단 코너
    if (y < 2400 and x >= 4200) or (x > 12400 and y <= 10600) or (x - y > 8000):
        return "BOT"                        # 아래쪽 가로 + 오른쪽 세로 띠 + 우하단 코너
    if abs(x - y) < 1600:
        return "MID"                        # 대각선 띠
    return "JUNGLE"                         # 정글 + 강가


ZONES = ["TOP", "MID", "BOT", "JUNGLE", "BASE"]

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

GAME_COLS = [
    "matchId", "gameVersion", "queueId", "gameDuration_s", "n_frames",
    "blueWin", "earlySurrender", "positionsValid",
]

PM_COLS = [
    "matchId", "minute", "timestamp_ms", "participantId", "teamId", "teamPosition",
    "puuid", "championName",
    # --- 프레임 스냅샷 (누적) ---
    "totalGold", "xp", "level", "minionsKilled", "jungleMinionsKilled",
    "dmgToChamp", "dmgTaken", "ccEnemy_ms", "pos_x", "pos_y",
    # --- 이벤트 누적 ---
    "kills", "deaths", "assists",
    "soloKills", "soloDeaths",          # 어시스트 없음 & 상대가 같은 teamPosition의 맞상대
    "teamKills",                         # 해당 시점까지 소속 팀 전체 킬 (killPart 계산용)
    "objKA", "teamObj",                  # 엘리트 몬스터 킬/어시스트, 팀 전체 엘리트 몬스터 처치
    "plates", "buildingKA",              # 포탑 방패 파괴, 건물 파괴 킬/어시스트
    "wardsPlaced", "wardsKilled",
    # --- 구역별 킬 관여(킬+어시스트) 누적: 로밍/갱킹 지표 계산용 ---
    "ka_TOP", "ka_MID", "ka_BOT", "ka_JUNGLE", "ka_BASE",
]

END_COLS = [
    "matchId", "participantId", "teamId", "teamPosition", "puuid", "win",
    "goldEarned", "kills", "deaths", "assists", "totalMinionsKilled",
    "neutralMinionsKilled", "visionScore", "wardsPlaced", "wardsKilled",
    "totalTimeCCDealt", "timeCCingOthers", "ch_soloKills", "ch_killParticipation",
]


P_KEEP = ["participantId", "teamId", "teamPosition", "puuid", "championName", "win",
          "gameEndedInEarlySurrender", "goldEarned", "kills", "deaths", "assists",
          "totalMinionsKilled", "neutralMinionsKilled", "visionScore", "wardsPlaced",
          "wardsKilled", "totalTimeCCDealt", "timeCCingOthers"]


def slim_match(g):
    """메모리 절약: match에서 필요한 필드만 남김 (1,000게임 이상 대비)."""
    info = g["info"]
    parts = []
    for p in info["participants"]:
        q = {k: p.get(k) for k in P_KEEP}
        ch = p.get("challenges", {}) or {}
        q["challenges"] = {"soloKills": ch.get("soloKills"), "killParticipation": ch.get("killParticipation")}
        parts.append(q)
    return {"metadata": {"matchId": g["metadata"]["matchId"]},
            "info": {"gameVersion": info.get("gameVersion"), "queueId": info.get("queueId"),
                     "gameDuration": info.get("gameDuration"), "teams": info.get("teams", []),
                     "participants": parts}}


def load_games(pattern, slim=False):
    games = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):  # 단일 게임 파일도 허용
            data = [data]
        for g in data:
            games[g["metadata"]["matchId"]] = slim_match(g) if slim else g
    return games


def parse_events(frames):
    """모든 이벤트를 timestamp 순으로 평탄화."""
    evs = [e for fr in frames for e in fr.get("events", [])]
    evs.sort(key=lambda e: e.get("timestamp", 0))
    return evs


def process_game(match, timeline, w_game, w_pm, w_end, warn):
    mid = match["metadata"]["matchId"]
    info = match["info"]
    parts = {p["participantId"]: p for p in info["participants"]}
    team_of = {pid: p["teamId"] for pid, p in parts.items()}
    pos_of = {pid: p.get("teamPosition", "") for pid, p in parts.items()}

    # 포지션 유효성: 팀마다 5개 포지션이 정확히 1명씩
    cnt = collections.Counter((team_of[pid], pos_of[pid]) for pid in parts)
    positions_valid = all(cnt.get((t, ps), 0) == 1 for t in (100, 200) for ps in POSITIONS)
    if not positions_valid:
        warn["invalid_positions"] += 1

    # 맞상대: 같은 teamPosition, 다른 팀
    opp = {}
    for a in parts:
        for b in parts:
            if team_of[a] != team_of[b] and pos_of[a] and pos_of[a] == pos_of[b]:
                opp[a] = b

    blue_win = next((t["win"] for t in info["teams"] if t["teamId"] == 100), None)
    early = any(p.get("gameEndedInEarlySurrender", False) for p in info["participants"])
    frames = timeline["info"]["frames"]

    w_game.writerow([mid, info.get("gameVersion"), info.get("queueId"), info.get("gameDuration"),
                     len(frames), int(bool(blue_win)), int(early), int(positions_valid)])

    # 이벤트 누적 카운터
    c = {pid: collections.Counter() for pid in parts}
    team_c = {100: collections.Counter(), 200: collections.Counter()}
    evs = parse_events(frames)
    ei = 0

    for minute, fr in enumerate(frames):
        ts = fr["timestamp"]
        # 이 프레임 시점까지의 이벤트 반영
        while ei < len(evs) and evs[ei].get("timestamp", 0) <= ts:
            e = evs[ei]
            ei += 1
            et = e["type"]
            if et == "CHAMPION_KILL":
                k, v = e.get("killerId", 0), e.get("victimId", 0)
                assists = e.get("assistingParticipantIds", []) or []  # 키 누락 대비
                if v in c:
                    c[v]["deaths"] += 1
                if k in c:  # killerId=0(처형 등)은 킬로 세지 않음
                    c[k]["kills"] += 1
                    team_c[team_of[k]]["kills"] += 1
                    if not assists and opp.get(k) == v:
                        c[k]["soloKills"] += 1
                        c[v]["soloDeaths"] += 1
                elif k == 0:
                    warn["kill_by_0"] += 1
                for a in assists:
                    if a in c:
                        c[a]["assists"] += 1
                # 킬이 일어난 구역별로 킬러+어시스트의 관여 횟수 누적
                ps = e.get("position") or {}
                zone = map_zone(ps.get("x"), ps.get("y"))
                for p in {k, *assists}:
                    if p in c:
                        c[p]["ka_" + zone] += 1
            elif et == "ELITE_MONSTER_KILL":
                k = e.get("killerId", 0)
                tid = e.get("killerTeamId")
                if tid in team_c:
                    team_c[tid]["obj"] += 1
                for p in {k, *(e.get("assistingParticipantIds", []) or [])}:
                    if p in c:
                        c[p]["objKA"] += 1
            elif et == "TURRET_PLATE_DESTROYED":
                k = e.get("killerId", 0)
                if k in c:
                    c[k]["plates"] += 1
            elif et == "BUILDING_KILL":
                for p in {e.get("killerId", 0), *(e.get("assistingParticipantIds", []) or [])}:
                    if p in c:
                        c[p]["buildingKA"] += 1
            elif et == "WARD_PLACED":
                p = e.get("creatorId", 0)
                if p in c:
                    c[p]["wardsPlaced"] += 1
                else:
                    warn["ward_creator_0"] += 1
            elif et == "WARD_KILL":
                p = e.get("killerId", 0)
                if p in c:
                    c[p]["wardsKilled"] += 1

        pframes = fr["participantFrames"]
        for pid, p in parts.items():
            q = pframes.get(str(pid))
            if q is None:
                warn["missing_pframe"] += 1
                continue
            ds = q.get("damageStats", {})
            pos = q.get("position", {}) or {}
            cc = c[pid]
            tc = team_c[team_of[pid]]
            w_pm.writerow([
                mid, minute, ts, pid, team_of[pid], pos_of[pid],
                p.get("puuid"), p.get("championName"),
                q.get("totalGold"), q.get("xp"), q.get("level"),
                q.get("minionsKilled"), q.get("jungleMinionsKilled"),
                ds.get("totalDamageDoneToChampions"), ds.get("totalDamageTaken"),
                q.get("timeEnemySpentControlled"), pos.get("x"), pos.get("y"),
                cc["kills"], cc["deaths"], cc["assists"],
                cc["soloKills"], cc["soloDeaths"], tc["kills"],
                cc["objKA"], tc["obj"], cc["plates"], cc["buildingKA"],
                cc["wardsPlaced"], cc["wardsKilled"],
                *[cc["ka_" + z] for z in ZONES],
            ])

    for pid, p in parts.items():
        ch = p.get("challenges", {}) or {}
        w_end.writerow([
            mid, pid, team_of[pid], pos_of[pid], p.get("puuid"), int(bool(p.get("win"))),
            p.get("goldEarned"), p.get("kills"), p.get("deaths"), p.get("assists"),
            p.get("totalMinionsKilled"), p.get("neutralMinionsKilled"), p.get("visionScore"),
            p.get("wardsPlaced"), p.get("wardsKilled"), p.get("totalTimeCCDealt"),
            p.get("timeCCingOthers"), ch.get("soloKills"), ch.get("killParticipation"),
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--match_glob", default="*match_*.json")
    ap.add_argument("--timeline_glob", default="*timeline_*.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 하위 폴더까지 검색 (압축 해제 시 폴더가 한 겹 더 생기는 경우 대비)
    match_files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.match_glob), recursive=True))
    tl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.timeline_glob), recursive=True))
    print(f"match 파일 {len(match_files)}개, timeline 파일 {len(tl_files)}개")
    if not match_files or not tl_files:
        print("파일을 찾지 못했습니다. --input_dir 경로와 파일명(match_*.json, timeline_*.json)을 확인하세요.")
        return

    def opener(name):
        return gzip.open(os.path.join(args.out_dir, name), "wt", newline="", encoding="utf-8")

    warn = collections.Counter()
    n_ok = 0
    with opener("games.csv.gz") as fg, opener("player_minute.csv.gz") as fp, opener("player_end.csv.gz") as fe:
        wg, wp, we = csv.writer(fg), csv.writer(fp), csv.writer(fe)
        wg.writerow(GAME_COLS); wp.writerow(PM_COLS); we.writerow(END_COLS)

        # 메모리 절약: timeline 파일을 하나씩 읽고, 대응하는 match는 전체 인덱스에서 찾음
        matches = {}
        print("match 파일 읽는 중...", flush=True)
        for path in match_files:
            matches.update(load_games(path, slim=True))
        print(f"match 게임 수: {len(matches)}")

        seen = set()
        t0 = time.time()
        for fi, path in enumerate(tl_files, 1):
            for mid, tl in load_games(path).items():
                if mid in seen:
                    warn["duplicate_timeline"] += 1
                    continue
                seen.add(mid)
                m = matches.get(mid)
                if m is None:
                    warn["timeline_without_match"] += 1
                    continue
                try:
                    process_game(m, tl, wg, wp, we, warn)
                    n_ok += 1
                except Exception as ex:  # 한 게임 오류로 전체가 멈추지 않도록
                    warn[f"error:{type(ex).__name__}"] += 1
            if fi % 20 == 0 or fi == len(tl_files):
                el = time.time() - t0
                eta = el / fi * (len(tl_files) - fi)
                print(f"  timeline {fi}/{len(tl_files)} 파일, 게임 {n_ok}개 처리, "
                      f"경과 {el/60:.1f}분, 남은 시간 약 {eta/60:.1f}분", flush=True)
        warn["match_without_timeline"] = len(set(matches) - seen)

    print(f"처리 완료 게임: {n_ok}")
    print("경고/점검 카운트:", dict(warn))


if __name__ == "__main__":
    main()