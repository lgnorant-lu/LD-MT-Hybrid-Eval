import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def run_ultimate_slang_heatmap():
    audit_root = Path("Local_PoLL_MQM/output/elite_five_integrated/audited_reports")
    output_dir = Path("Local_PoLL_MQM/analysis_infra/raw_stats")
    output_plots = Path("Local_PoLL_MQM/analysis_infra/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. 提取所有模型的全量 S_final (最终得分)
    # S_final 已经综合了：
    # - 7个裁判的 MQM 语义共识 (avg_s_mqm)
    # - 术语门控处罚 (term_gate penalty)
    # - 客观错误惩罚 (p_obj)
    
    all_models = [d.name for d in audit_root.glob("*") if d.is_dir()]
    slang_matrix = defaultdict(dict)
    
    print(f"Extracting S_final from {len(all_models)} models for 207 slang items...")
    
    for model_id in all_models:
        slang_report = audit_root / model_id / "Slang_Ambiguous_poll_mqm_audit.json"
        if not slang_report.exists():
            print(f"Warning: Missing report for {model_id}")
            continue
            
        with open(slang_report, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for res in data.get("results", []):
            source = res.get("source", "")
            # 提取标题作为唯一标识
            item_label = source.split("\n")[0].replace("Title: ", "").strip()[:30]
            
            # 核心：使用 s_final 作为最终衡量维度 (0-100)
            # 如果 s_final 不存在，尝试从 audit_score 获取
            s_final = res.get("s_final", 0.0)
            slang_matrix[item_label][model_id] = s_final

    if not slang_matrix:
        print("Error: No slang matrix data found.")
        return

    # 2. 转换为 DataFrame
    df = pd.DataFrame(slang_matrix).T.fillna(0.0)
    
    # 清洗模型列名
    df.columns = [c.replace("Qwen--", "").replace("google--", "").replace("tencent--", "HY-").replace("CohereLabs--", "") for c in df.columns]
    
    # 按全模型平均表现排序，让图表呈现清晰的梯度
    df['avg_perf'] = df.mean(axis=1)
    df = df.sort_values('avg_perf', ascending=False)
    df_plot = df.drop(columns=['avg_perf'])
    
    # 3. 保存全量统计数据
    df_plot.to_csv(output_dir / "slang_s_final_matrix.csv", encoding='utf-8-sig')
    
    # 4. 绘图：极致深度的消歧生存矩阵
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 动态高度：207 项需要足够的高度
    plot_height = max(15, len(df_plot) * 0.25)
    plt.figure(figsize=(18, plot_height))
    
    # 使用极致对比的调色盘：深红 (0) -> 橙 -> 黄 -> 深绿 (100)
    # 这种色阶能一眼看出模型在哪里“全军覆没”，在哪里“平分秋色”
    custom_colors = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"] 
    cmap = mcolors.LinearSegmentedColormap.from_list("poll_mqm_final", custom_colors, N=100)

    ax = sns.heatmap(df_plot, 
                cmap=cmap, 
                vmin=0, vmax=100,
                annot=False, 
                cbar_kws={'label': '消歧终极得分 (S_final: 语义+术语融合)', 'ticks': [0, 20, 40, 60, 80, 100]},
                linewidths=0.1,
                linecolor='#eeeeee')

    plt.title(f"L-Station 极客消歧全景矩阵 (全量 {len(df_plot)} 项：语义保真 + 术语门控)", 
              fontsize=22, fontweight='bold', pad=30)
    plt.xlabel("翻译模型 (Stage 2: 五大精英仲裁团评审结果)", fontsize=15, labelpad=20)
    plt.ylabel("语料标题 (按全模型平均分降序排列)", fontsize=15, labelpad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plot_path = output_plots / "06_Slang_Disambiguation_S_Final_Heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Ultimate Disambiguation Heatmap saved to {plot_path}")

if __name__ == "__main__":
    run_ultimate_slang_heatmap()
