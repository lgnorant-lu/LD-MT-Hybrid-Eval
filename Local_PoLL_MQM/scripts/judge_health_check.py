import json
from pathlib import Path
from collections import defaultdict

def judge_health_check():
    root = Path("output/full_seven_integrated/checkpoints")
    if not root.exists():
        print("Root not found.")
        return

    # judge -> {total_calls, success_calls, failure_calls}
    health_map = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})
    
    print(f"Deep scanning 7 judges health status...")

    for model_dir in root.iterdir():
        if not model_dir.is_dir(): continue
        for json_file in model_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except: continue
                
                for tid, judges in data.items():
                    for jid, decision in judges.items():
                        health_map[jid]["total"] += 1
                        if decision.get("ok", False):
                            health_map[jid]["success"] += 1
                        else:
                            health_map[jid]["fail"] += 1

    print("\n" + "="*60)
    print(f"{'Judge ID':<25} | {'Success%':<10} | {'Success':<8} | {'Fail':<8}")
    print("-" * 60)
    
    for jid, stats in sorted(health_map.items()):
        total = stats["total"]
        success = stats["success"]
        fail = stats["fail"]
        rate = (success / total * 100) if total > 0 else 0
        print(f"{jid:<25} | {rate:>8.2f}% | {success:<8} | {fail:<8}")
    print("="*60)

if __name__ == "__main__":
    judge_health_check()
