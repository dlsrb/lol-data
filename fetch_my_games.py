"""
Riot API로 내 솔로랭크 경기(match + timeline) 받기

준비
  1) https://developer.riotgames.com 에 로그인해 개발용 API 키(24시간 유효)를 발급받는다.
  2) 키를 환경 변수 RIOT_API_KEY에 넣거나 --key로 넘긴다. (키는 코드나 깃허브에 올리지 않는다)

사용 예
  python fetch_my_games.py --riot_id "게임이름#KR1" --count 20 --key RGAPI-xxxx
  → my_match.json, my_timeline.json 저장 → my_mci.py로 MCI 계산

참고: 한국 서버 계정의 Account-V1, Match-V5는 asia 라우팅을 쓴다.
      개발용 키는 2분에 100번까지 요청할 수 있어, 요청 사이에 1.3초씩 쉰다.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://asia.api.riotgames.com"


def get(path, key):
    req = urllib.request.Request(BASE + path, headers={"X-Riot-Token": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                time.sleep(1.3)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # 요청 한도 초과 → 잠시 기다렸다 다시
                time.sleep(int(e.headers.get("Retry-After", 10)) + 1)
                continue
            raise SystemExit(f"API 오류 {e.code}: {path}  (키가 만료됐거나 Riot ID가 틀렸는지 확인)")
    raise SystemExit("요청 한도 초과가 계속됩니다. 잠시 후 다시 실행하세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--riot_id", required=True, help='예: "게임이름#KR1"')
    ap.add_argument("--count", type=int, default=20, help="가져올 최근 솔로랭크 경기 수 (최대 100)")
    ap.add_argument("--key", default=os.environ.get("RIOT_API_KEY", ""))
    ap.add_argument("--out_prefix", default="my")
    a = ap.parse_args()
    if not a.key:
        raise SystemExit("API 키가 없습니다. --key 또는 환경 변수 RIOT_API_KEY를 지정하세요.")
    name, tag = a.riot_id.split("#")
    acc = get(f"/riot/account/v1/accounts/by-riot-id/{urllib.parse.quote(name)}/{urllib.parse.quote(tag)}", a.key)
    ids = get(f"/lol/match/v5/matches/by-puuid/{acc['puuid']}/ids?queue=420&start=0&count={a.count}", a.key)
    print(f"{a.riot_id}: 솔로랭크 {len(ids)}경기")
    matches, timelines = [], []
    for i, mid in enumerate(ids, 1):
        matches.append(get(f"/lol/match/v5/matches/{mid}", a.key))
        timelines.append(get(f"/lol/match/v5/matches/{mid}/timeline", a.key))
        print(f"  {i}/{len(ids)} {mid}")
    for kind, data in (("match", matches), ("timeline", timelines)):
        with open(f"{a.out_prefix}_{kind}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    print(f"저장: {a.out_prefix}_match.json, {a.out_prefix}_timeline.json")
    print(f'다음 단계: python my_mci.py --match {a.out_prefix}_match.json --timeline {a.out_prefix}_timeline.json --name "{name}"')


if __name__ == "__main__":
    main()
