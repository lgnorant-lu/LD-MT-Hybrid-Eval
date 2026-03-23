import json
from pathlib import Path
from collections import defaultdict

def final_four_audit():
    root = Path("output/full_seven_integrated/checkpoints")
    targets = ['gpt_5_01', 'claude_4_6_01', 'qwen_3_5_sf', 'minimax_m2_5_sf']
    
    # Statistics
    stats = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0, "with_errors": 0})
    
    print(f"Auditing Elite Four Judges in: {root}\n")

    for json_file in root.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: continue
        
        for tid, judges in data.items():
            for jid in targets:
                if jid not in judges:
                    continue
                
                decision = judges[jid]
                stats[jid]["total"] += 1
                if decision.get("ok", False):
                    stats[jid]["ok"] += 1
                    if decision.get("errors") and len(decision.get("errors")) > 0:
                        stats[jid]["with_errors"] += 1
                else:
                    stats[jid]["fail"] += 1

    print("="*75)
    print(f"{'Judge ID':<20} | {'Total':<6} | {'OK':<6} | {'Fail':<6} | {'ErrFound':<10} | {'Success%'}")
    print("-" * 75)
    
    for jid in targets:
        s = stats[jid]
        total = s["total"]
        if total == 0:
            print(f"{jid:<20} | NO DATA FOUND")
            continue
        success_rate = (s["ok"] / total) * 100
        print(f"{jid:<20} | {total:<6} | {s['ok']:<6} | {s['fail']:<6} | {s['with_errors']:<10} | {success_rate:>7.2f}%")
    print("="*75)
    
    # Cross-check: How many items have all 4 judges ok?
    all_four_ok_count = 0
    total_unique_items = 0
    for json_file in root.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue
            for tid, judges in data.items():
                total_unique_items += 1
                if all(jid in judges and judges[jid].get("ok") for jid in targets):
                    all_four_ok_count += 1
    
    print(f"\nConsensus Potential:")
    print(f"  - Total Unique Test Items: {total_unique_items}")
    print(f"  - Items with ALL 4 Judges OK: {all_four_ok_count} ({(all_four_ok_count/total_unique_items*100):.2f}%)")

if __name__ == "__main__":
    final_four_audit()
