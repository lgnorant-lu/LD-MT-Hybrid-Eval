import json
from pathlib import Path
from collections import defaultdict, Counter

def elite_five_audit():
    root = Path("output/elite_five_integrated/checkpoints")
    targets = ['gpt_5_01', 'claude_4_6_01', 'qwen_3_5_sf', 'minimax_m2_5_sf', 'deepseek_v3_2_sf']
    
    # 1. Health Stats
    health = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0})
    # 2. Consensus Stats: How many valid judges per test item
    consensus_distribution = Counter()
    
    total_unique_items = 0

    print(f"Deep Auditing Elite Five Judges in: {root}\n")

    for json_file in root.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: continue
        
        for tid, judges in data.items():
            total_unique_items += 1
            valid_judges_count = 0
            
            for jid in targets:
                if jid not in judges:
                    continue
                
                decision = judges[jid]
                health[jid]["total"] += 1
                if decision.get("ok", False):
                    health[jid]["ok"] += 1
                    valid_judges_count += 1
                else:
                    health[jid]["fail"] += 1
            
            consensus_distribution[valid_judges_count] += 1

    print("="*70)
    print(f"{'Judge ID':<20} | {'Success%':<10} | {'Success':<8} | {'Fail':<8}")
    print("-" * 70)
    
    for jid in sorted(targets):
        s = health[jid]
        total = s["total"]
        if total == 0:
            print(f"{jid:<20} | NO DATA")
            continue
        rate = (s["ok"] / total) * 100
        print(f"{jid:<20} | {rate:>8.2f}% | {s['ok']:<8} | {s['fail']:<8}")
    print("="*70)
    
    print(f"\nConsensus Depth (Valid Judges per Item):")
    print(f"Total Unique Evaluation Items: {total_unique_items}")
    for i in range(len(targets) + 1):
        count = consensus_distribution[i]
        perc = (count / total_unique_items * 100) if total_unique_items > 0 else 0
        status = "✅ ROBUST" if i >= 3 else "❌ WEAK"
        print(f"  - {i} Judges OK: {count:<6} ({perc:>5.2f}%) [{status}]")

if __name__ == "__main__":
    elite_five_audit()
