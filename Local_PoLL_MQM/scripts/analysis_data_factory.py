import json
import csv
from pathlib import Path
from collections import defaultdict

def build_data_factory():
    audit_root = Path("Local_PoLL_MQM/output/elite_five_integrated/audited_reports")
    leaderboard_file = Path("Local_PoLL_MQM/output/elite_five_integrated/leaderboard/Global_PoLL_MQM_Summary.json")
    out_dir = Path("Local_PoLL_MQM/analysis_infra/dim_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not leaderboard_file.exists():
        print(f"Error: {leaderboard_file} not found.")
        return

    with open(leaderboard_file, "r", encoding="utf-8") as f:
        lb_data = json.load(f)

    global_cap = []
    err_topo = defaultdict(int) # (model, block, category, severity) -> count
    slang_matrix = [] # model, term, status (1/0), test_id
    judge_consensus = {
        "vote_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
        "judge_stats": defaultdict(lambda: {"proposed": 0, "accepted": 0})
    }
    divergence_cases = []

    for entry in lb_data.get("models", []):
        model_id = entry["model_id"]
        folder = entry["model_folder"]

        model_path = audit_root / folder
        if not model_path.exists(): continue

        for rep_file in model_path.glob("*.json"):
            with open(rep_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("audit_meta", {})
            block = meta.get("block", "Unknown")
            summary = data.get("block_summary", {})

            # 1. Global Capability Data
            global_cap.append({
                "model_id": model_id,
                "block": block,
                "avg_s_final": summary.get("avg_s_final", 0),
                "avg_s_mqm": summary.get("avg_s_mqm", 0),
                "avg_chrf": summary.get("avg_chrf_score", 0),
                "avg_comet": summary.get("avg_comet_score", 0),
                "avg_p_obj": summary.get("avg_p_obj", 0)
            })

            # Process individual test results
            for res in data.get("results", []):
                tid = res.get("test_id", "")

                # 2. Slang / Terminology Matrix
                tg = res.get("term_gate", {})
                if tg and tg.get("active"):
                    for t in tg.get("expected_hits", []):
                        slang_matrix.append({"model_id": model_id, "test_id": tid, "term": t, "hit": 1})
                    for t in tg.get("missing_expected", []):
                        slang_matrix.append({"model_id": model_id, "test_id": tid, "term": t, "hit": 0})

                # 3 & 4 & 5. Error Topology, Consensus, Divergence
                for err in res.get("accepted_errors", []):
                    cat = err.get("category", "other")
                    sev = err.get("final_severity", "minor")
                    votes = err.get("votes", 0)
                    j_ids = err.get("judge_ids", [])

                    # Topology
                    err_topo[(model_id, block, cat, sev)] += 1

                    # Consensus - Distribution
                    v_str = str(votes)
                    if v_str in judge_consensus["vote_distribution"]:
                        judge_consensus["vote_distribution"][v_str] += 1

                    # Consensus - Judge Level
                    for j in j_ids:
                        judge_consensus["judge_stats"][j]["proposed"] += 1
                        judge_consensus["judge_stats"][j]["accepted"] += 1

                    # Divergence (Marginal Pass: exactly 3 votes)
                    if votes == 3:
                        divergence_cases.append({
                            "model_id": model_id, "block": block, "test_id": tid,
                            "type": "marginal_pass", "span": err.get("span"),
                            "category": cat, "severity": sev,
                            "votes": votes, "judges": j_ids, "reasons": err.get("reasons", []),
                            "source": res.get("source"), "hypothesis": res.get("hypothesis")
                        })

                for err in res.get("rejected_errors", []):
                    votes = err.get("votes", 0)
                    j_ids = err.get("judge_ids", [])

                    v_str = str(votes)
                    if v_str in judge_consensus["vote_distribution"]:
                        judge_consensus["vote_distribution"][v_str] += 1

                    for j in j_ids:
                        judge_consensus["judge_stats"][j]["proposed"] += 1
                        # not accepted

                    # Divergence (Marginal Fail: exactly 2 votes)
                    if votes == 2:
                        divergence_cases.append({
                            "model_id": model_id, "block": block, "test_id": tid,
                            "type": "marginal_fail", "span": err.get("span"),
                            "category": err.get("category", "other"), "severity": "rejected",
                            "votes": votes, "judges": j_ids, "reasons": [],
                            "source": res.get("source"), "hypothesis": res.get("hypothesis")
                        })

    # Write Outputs
    with open(out_dir / "dim_global_capability.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "block", "avg_s_final", "avg_s_mqm", "avg_chrf", "avg_comet", "avg_p_obj"])
        writer.writeheader()
        writer.writerows(global_cap)

    with open(out_dir / "dim_error_topology.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "block", "category", "severity", "count"])
        for (m, b, c, s), count in err_topo.items():
            writer.writerow([m, b, c, s, count])

    with open(out_dir / "dim_slang_matrix.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "test_id", "term", "hit"])
        writer.writeheader()
        writer.writerows(slang_matrix)

    # Convert default dict to normal dict before serialization
    judge_consensus["judge_stats"] = dict(judge_consensus["judge_stats"])
    for j in judge_consensus["judge_stats"]:
        props = judge_consensus["judge_stats"][j]["proposed"]
        acc = judge_consensus["judge_stats"][j]["accepted"]
        judge_consensus["judge_stats"][j]["reliability_rate"] = round(acc / props * 100, 2) if props > 0 else 0

    with open(out_dir / "dim_judge_consensus.json", "w", encoding="utf-8") as f:
        json.dump(judge_consensus, f, indent=2)

    with open(out_dir / "dim_divergence_cases.json", "w", encoding="utf-8") as f:
        json.dump(divergence_cases, f, indent=2, ensure_ascii=False)

    print("Stage 3 Data Factory executed successfully.")
    print("Primitives generated in: Local_PoLL_MQM/analysis_infra/dim_data/")

if __name__ == "__main__":
    build_data_factory()
