"""Check a leaderboard CSV (scripts/evaluate_leaderboard.py) against our competition result file, flight by flight.

    python bluesky_rl_transformer/analysis/leaderboard_check.py leaderboard_sa.csv \
        bluesky_rl_transformer/results/sa_tf_info_ff_cls_big_s0_competition_sa.json

Prints the number of flights where all 8 metrics are equal, and the leaderboard summary (clean flights %, goal %,
events per flight, flight time) with the ranking rule of the Space.
"""
import csv
import json
import sys

KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time", "restricted_area_events",
        "time_in_restricted_area", "sector_exit_events", "time_outside_sector"]

csv_path, json_path = sys.argv[1], sys.argv[2]
lb = {int(r["episode"]): r for r in csv.DictReader(open(csv_path))}
ours = {int(r["episode_index"]): r for r in json.load(open(json_path))["rows"]}
same = [e for e in lb if e in ours and all(abs(float(lb[e][k]) - float(ours[e][k])) < 1e-6 for k in KEYS)]
print(f"{len(lb)} CSV flights, {len(ours)} result flights, all 8 metrics equal in {len(same)} flights")
diff = sorted(set(lb) - set(same))[:10]
for e in diff:
    print(f"  episode {e}: csv {[lb[e][k] for k in KEYS]} | ours {[ours.get(e, {}).get(k) for k in KEYS]}")
n = len(lb)
m = {k: sum(float(r[k]) for r in lb.values()) / n for k in KEYS}
clean = sum(1 for r in lb.values() if float(r["waypoint_reached"]) == 1 and float(r["intrusion_events"]) == 0
            and float(r["restricted_area_events"]) == 0 and float(r["sector_exit_events"]) == 0) / n * 100
print(f"clean {clean:.1f} %, goal {100 * m['waypoint_reached']:.1f} %, intrusions {m['intrusion_events']:.3f}, "
      f"area {m['restricted_area_events']:.3f}, sector {m['sector_exit_events']:.3f}, flight {m['flight_time']:.1f} s")
