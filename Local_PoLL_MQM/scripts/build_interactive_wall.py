import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def build_interactive_wall():
    data_dir = Path("Local_PoLL_MQM/analysis_infra/dim_data")
    out_html = Path("Local_PoLL_MQM/analysis_infra/Interactive_Compare_Wall_V2.html")

    # Load Data
    try:
        df_global = pd.read_csv(data_dir / "dim_global_capability.csv")
        global_data = df_global.to_dict(orient="records")
    except Exception:
        global_data = []

    try:
        df_topo = pd.read_csv(data_dir / "dim_error_topology.csv")
        topo_data = df_topo.to_dict(orient="records")
    except Exception:
        topo_data = []

    try:
        df_slang = pd.read_csv(data_dir / "dim_slang_matrix.csv")
        slang_data = df_slang.to_dict(orient="records")
    except Exception:
        slang_data = []

    try:
        with open(data_dir / "dim_divergence_cases.json", "r", encoding="utf-8") as f:
            divergence_data = json.load(f)
    except Exception:
        divergence_data = []

    try:
        with open(data_dir / "dim_judge_consensus.json", "r", encoding="utf-8") as f:
            judge_data = json.load(f)
    except Exception:
        judge_data = {}

    # Build V1-style compare wall data from audited_reports
    audit_root = Path("Local_PoLL_MQM/output/elite_five_integrated/audited_reports")
    leaderboard_file = Path(
        "Local_PoLL_MQM/output/elite_five_integrated/leaderboard/Global_PoLL_MQM_Summary.json"
    )
    folder_to_id = {}
    if leaderboard_file.exists():
        try:
            lb = json.loads(leaderboard_file.read_text(encoding="utf-8"))
            folder_to_id = {
                m["model_folder"]: m["model_id"] for m in lb.get("models", [])
            }
        except Exception:
            folder_to_id = {}

    master_data = defaultdict(lambda: {"source": "", "reference": "", "models": {}})
    all_models = []

    for model_dir in audit_root.glob("*"):
        if not model_dir.is_dir():
            continue
        model_folder = model_dir.name
        model_id = folder_to_id.get(model_folder, model_folder)
        all_models.append(model_id)

        for report_file in model_dir.glob("*.json"):
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            for res in report.get("results", []):
                tid = res.get("test_id", "")
                if not master_data[tid]["source"]:
                    master_data[tid]["source"] = (
                        res.get("source") or res.get("source_text") or "N/A"
                    )
                if (
                    not master_data[tid]["reference"]
                    or master_data[tid]["reference"] == "N/A"
                ):
                    master_data[tid]["reference"] = (
                        res.get("reference") or res.get("reference_text") or "N/A"
                    )

                master_data[tid]["models"][model_id] = {
                    "hyp": res.get("hypothesis") or res.get("hypothesis_text") or "N/A",
                    "errors": res.get("accepted_errors", []),
                }

    all_models = sorted(set(all_models))

    cards = []
    for tid, data in master_data.items():
        model_rows = []
        for mid in all_models:
            m_res = data["models"].get(mid, {"hyp": "N/A", "errors": []})
            err_html = ""
            if not m_res["errors"]:
                err_html = '<span class="text-success" style="font-weight:500;">通过 (无共识错误)</span>'
            else:
                for err in m_res["errors"]:
                    sev_raw = str(
                        err.get("final_severity", err.get("severity", "minor"))
                    ).lower()
                    reasons = err.get("reasons") or err.get("reason") or []
                    if isinstance(reasons, list):
                        reason_text = " | ".join(
                            list({str(r).strip() for r in reasons if r})
                        )
                    else:
                        reason_text = str(reasons)

                    if not reason_text or reason_text.lower() == "none":
                        reason_text = "未提供具体理由"

                    sev_map = {"minor": "轻微", "major": "严重", "critical": "致命"}
                    cat_map = {
                        "accuracy": "准确性",
                        "fluency": "流畅性",
                        "terminology": "术语",
                        "style": "风格",
                        "other": "其他",
                    }
                    sev_cn = sev_map.get(sev_raw, "轻微")
                    cat_cn = cat_map.get(
                        str(err.get("category", "other")).lower(), "其他"
                    )

                    err_html += f"""
                    <div class="mb-2">
                        <span class="error-tag sev-{sev_raw}">{sev_cn}</span>
                        <small class="text-secondary">[{cat_cn}]</small>
                        <span class="text-danger" style="font-size:0.9rem;">"{err.get("span", "")}"</span>
                        <br><small class="text-muted">理由: {reason_text}</small>
                    </div>
                    """

            model_rows.append(
                f"""
                            <tr>
                                <td><small><strong>{mid}</strong></small></td>
                                <td>{m_res["hyp"]}</td>
                                <td>{err_html}</td>
                            </tr>
                """
            )

        cards.append(
            f"""
        <div class="card item-card" data-id="{tid}" data-search="{tid} {data["source"]}">
            <div class="item-header">
                <h5 class="m-0">测试案例 ID: {tid}</h5>
            </div>
            <div class="card-body">
                <div class="row mb-4">
                    <div class="col-md-6">
                        <span class="label-title">原始语料 (SOURCE)</span>
                        <div class="source-box">{data["source"]}</div>
                    </div>
                    <div class="col-md-6">
                        <span class="label-title">标准参考译文 (GOLDEN REFERENCE)</span>
                        <div class="reference-box">{data["reference"]}</div>
                    </div>
                </div>

                <span class="label-title">15 个模型译文与 3/5 多数共识仲裁结果</span>
                <div class="table-responsive">
                    <table class="table table-hover align-middle model-table">
                        <thead class="table-dark">
                            <tr>
                                <th style="width: 180px;">待测模型</th>
                                <th>机器翻译结果 (Hypothesis)</th>
                                <th>仲裁结论 (Accepted Errors)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(model_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        )

    compare_wall_html = "\n".join(cards)

    # Pack into a single JSON object for the frontend
    embedded_data = {
        "global": global_data,
        "topology": topo_data,
        "slang": slang_data,
        "divergence": divergence_data,
        "judge_consensus": judge_data,
    }

    data_json_str = json.dumps(embedded_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L-Station Interactive Compare Wall V2</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{ background-color: #f4f7f6; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; }}
        .nav-pills .nav-link.active {{ background-color: #1a1c23; }}
        .card {{ border: none; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom: 20px; }}
        .chart-container {{ height: 430px; width: 100%; }}
        .sidebar {{ background-color: #fff; min-height: 100vh; padding: 20px; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }}
        .item-header {{ background-color: #1a1c23; color: #ffffff; padding: 1.25rem; border-bottom: 1px solid #2d2f39; }}
        .source-box {{ background-color: #ffffff; padding: 1.25rem; border-left: 5px solid #007bff; border-radius: 4px; font-size: 0.95rem; }}
        .reference-box {{ background-color: #ffffff; padding: 1.25rem; border-left: 5px solid #28a745; border-radius: 4px; font-size: 0.95rem; }}
        .model-table {{ margin-top: 1.5rem; border-collapse: separate; border-spacing: 0; }}
        .error-tag {{ font-size: 0.75rem; padding: 0.15rem 0.6rem; border-radius: 4px; margin-right: 0.5rem; font-weight: bold; }}
        .sev-minor {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        .sev-major {{ background-color: #ffe5d0; color: #d35400; border: 1px solid #fadbd8; }}
        .sev-critical {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .search-bar {{ position: sticky; top: 0; z-index: 1000; background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); padding: 1.5rem; box-shadow: 0 2px 15px rgba(0,0,0,0.1); }}
        .label-title {{ font-weight: bold; color: #4a4a4a; margin-bottom: 0.5rem; display: block; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem; }}
        .select-compact {{ max-width: 360px; }}
        .muted-note {{ font-size: 0.9rem; color: #6c757d; }}
        .chip {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; background: #f1f1f1; font-size: 0.75rem; margin-right: 0.4rem; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar Navigation -->
            <div class="col-md-2 sidebar">
                <h4 class="mb-4">L-Station V2</h4>
                <div class="nav flex-column nav-pills" id="v-pills-tab" role="tablist" aria-orientation="vertical">
                    <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab-compare" type="button" role="tab">Compare Wall</button>
                    <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-dashboard" type="button" role="tab">Leaderboard</button>
                    <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-arena" type="button" role="tab">1vN Arena</button>
                    <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-slang" type="button" role="tab">Slang Court</button>
                    <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-divergence" type="button" role="tab">Divergence</button>
                    <button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-audit" type="button" role="tab">Audit (Judges)</button>
                </div>
            </div>

            <!-- Main Content Area -->
            <div class="col-md-10 p-4">
                <div class="tab-content" id="v-pills-tabContent">

                    <!-- Compare Wall -->
                    <div class="tab-pane fade show active" id="tab-compare" role="tabpanel">
                        <div class="search-bar">
                            <div class="container">
                                <h3 class="mb-3">L-Station 机器翻译全量对比墙（五大精英评审团共识版）</h3>
                                <input type="text" id="searchInput" class="form-control form-control-lg" placeholder="搜索测试项 ID 或关键词 (例如: 反代, 降智, 始皇)...">
                            </div>
                        </div>
                        <div id="contentWall" class="mt-4">
                            {compare_wall_html}
                        </div>
                    </div>

                    <!-- Dashboard -->
                    <div class="tab-pane fade" id="tab-dashboard" role="tabpanel">
                        <h2>Global Leaderboard</h2>
                        <div class="card p-3">
                            <div class="d-flex gap-2 align-items-center mb-2">
                                <span class="label-title m-0">Metric</span>
                                <select id="metricSelect" class="form-select form-select-sm select-compact"></select>
                                <span class="label-title m-0">Block</span>
                                <select id="leaderboardBlockSelect" class="form-select form-select-sm select-compact"></select>
                            </div>
                            <div id="chart-leaderboard" class="chart-container"></div>
                        </div>
                    </div>

                    <!-- Arena -->
                    <div class="tab-pane fade" id="tab-arena" role="tabpanel">
                        <h2>1vN Arena</h2>
                        <div class="card p-3">
                            <div class="d-flex flex-wrap gap-2 align-items-center mb-2">
                                <span class="label-title m-0">Baseline</span>
                                <select id="baselineSelect" class="form-select form-select-sm select-compact"></select>
                                <span class="label-title m-0">Block</span>
                                <select id="arenaBlockSelect" class="form-select form-select-sm select-compact"></select>
                            </div>
                            <div class="muted-note mb-2">展示每个模型相对基线的平均 S_Final 差值（越高越好）。</div>
                            <div id="chart-arena" class="chart-container"></div>
                        </div>
                    </div>

                    <!-- Slang -->
                    <div class="tab-pane fade" id="tab-slang" role="tabpanel">
                        <h2>Slang Disambiguation Court</h2>
                        <div class="card p-3">
                            <div class="muted-note mb-2">术语/黑话命中率（term-gate hit rate）</div>
                            <div id="chart-slang" class="chart-container"></div>
                            <div class="mt-3">
                                <div class="label-title">Top Terms (Lowest Hit Rate)</div>
                                <div id="slang-terms" class="mt-2"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Divergence -->
                    <div class="tab-pane fade" id="tab-divergence" role="tabpanel">
                        <h2>Controversial Divergence Cases</h2>
                        <div class="card p-3">
                            <div class="d-flex gap-2 align-items-center mb-2">
                                <span class="label-title m-0">Filter</span>
                                <input id="divergenceSearch" class="form-control form-control-sm select-compact" placeholder="model_id / test_id / block">
                            </div>
                            <div id="divergence-container">
                                <p>Loading divergence cases...</p>
                            </div>
                        </div>
                    </div>

                    <!-- Audit -->
                    <div class="tab-pane fade" id="tab-audit" role="tabpanel">
                        <h2>Judge Consensus Audit</h2>
                        <div class="card p-3" id="audit-container">
                            <p>For diagnostic use only.</p>
                            <pre id="audit-json" style="max-height: 500px; overflow-y: auto; background: #eee; padding: 10px;"></pre>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const DB = {data_json_str};

        let leaderboardChart = null;
        let arenaChart = null;
        let slangChart = null;
        let divergenceItems = [];

        function truncateLabel(label, maxLen = 24) {{
            const s = String(label || '');
            if (s.length <= maxLen) return s;
            return s.slice(0, maxLen - 3) + '...';
        }}

        function uniqueBlocks() {{
            const set = new Set();
            DB.global.forEach(r => set.add(r.block));
            return Array.from(set).filter(Boolean).sort();
        }}

        function computeScores(metricKey, block) {{
            const sums = {{}};
            const counts = {{}};
            DB.global.forEach(r => {{
                if (block && block !== 'Overall' && r.block !== block) return;
                const m = r.model_id;
                if (!sums[m]) {{ sums[m] = 0; counts[m] = 0; }}
                sums[m] += Number(r[metricKey] || 0);
                counts[m] += 1;
            }});
            const avg = {{}};
            Object.keys(sums).forEach(m => {{
                avg[m] = counts[m] ? (sums[m] / counts[m]) : 0;
            }});
            return avg;
        }}

        function initLeaderboard() {{
            const el = document.getElementById('chart-leaderboard');
            if (!el) return;
            leaderboardChart = echarts.init(el);

            const metricSelect = document.getElementById('metricSelect');
            const blockSelect = document.getElementById('leaderboardBlockSelect');

            const metricOptions = [
                {{ label: 'S_Final', key: 'avg_s_final', max: 100 }},
                {{ label: 'S_MQM', key: 'avg_s_mqm', max: 100 }},
                {{ label: 'COMET', key: 'avg_comet', max: 100 }},
            ];

            metricSelect.innerHTML = metricOptions.map(m => `<option value="${{m.key}}">${{m.label}}</option>`).join('');
            metricSelect.value = 'avg_s_final';

            const blocks = ['Overall', ...uniqueBlocks()];
            blockSelect.innerHTML = blocks.map(b => `<option value="${{b}}">${{b}}</option>`).join('');
            blockSelect.value = 'Overall';

            function render() {{
                const metricKey = metricSelect.value;
                const metricLabel = metricOptions.find(m => m.key === metricKey)?.label || metricKey;
                const maxVal = metricOptions.find(m => m.key === metricKey)?.max || 100;
                const block = blockSelect.value;

                const avg = computeScores(metricKey, block);
                const rows = Object.keys(avg).map(m => ({{ m, v: avg[m] }})).sort((a,b) => b.v - a.v);
                const models = rows.map(r => r.m);
                const values = rows.map(r => Number(r.v.toFixed(2)));

                const option = {{
                    title: {{ text: `${{metricLabel}} Global Ranking (${{
                        block
                    }})` }},
                    tooltip: {{
                        trigger: 'axis',
                        axisPointer: {{ type: 'shadow' }},
                        formatter: (params) => {{
                            const p = params[0];
                            return `${{p.name}}<br/>${{metricLabel}}: ${{p.value}}`;
                        }}
                    }},
                    grid: {{ left: 220, right: 30, bottom: 20, containLabel: true }},
                    xAxis: {{ type: 'value', max: maxVal }},
                    yAxis: {{
                        type: 'category',
                        data: models,
                        axisLabel: {{
                            interval: 0,
                            formatter: (v) => truncateLabel(v, 26)
                        }}
                    }},
                    series: [{{
                        name: metricLabel,
                        type: 'bar',
                        data: values,
                        label: {{ show: true, position: 'right' }},
                        itemStyle: {{ color: '#5470c6' }}
                    }}]
                }};
                leaderboardChart.setOption(option);
            }}

            metricSelect.addEventListener('change', render);
            blockSelect.addEventListener('change', render);
            render();
        }}

        function initArena() {{
            const el = document.getElementById('chart-arena');
            if (!el) return;
            arenaChart = echarts.init(el);

            const baselineSelect = document.getElementById('baselineSelect');
            const blockSelect = document.getElementById('arenaBlockSelect');

            const blocks = ['Overall', ...uniqueBlocks()];
            blockSelect.innerHTML = blocks.map(b => `<option value="${{b}}">${{b}}</option>`).join('');
            blockSelect.value = 'Overall';

            function render() {{
                const block = blockSelect.value;
                const avg = computeScores('avg_s_final', block);
                const models = Object.keys(avg).sort((a, b) => avg[b] - avg[a]);

                baselineSelect.innerHTML = models.map(m => `<option value="${{m}}">${{m}}</option>`).join('');
                if (!baselineSelect.value || !avg[baselineSelect.value]) {{
                    baselineSelect.value = models[0] || '';
                }}

                const baseline = baselineSelect.value;
                if (!baseline || !avg[baseline]) return;

                const baseScore = avg[baseline];
                const deltas = models.map(m => ({{ m, d: avg[m] - baseScore }}));
                deltas.sort((a,b) => b.d - a.d);

                const axisModels = deltas.map(x => x.m);
                const axisValues = deltas.map(x => Number(x.d.toFixed(2)));

                const option = {{
                    title: {{ text: `Δ S_Final vs Baseline: ${{truncateLabel(baseline, 40)}}` }},
                    tooltip: {{
                        trigger: 'axis',
                        axisPointer: {{ type: 'shadow' }},
                        formatter: (params) => {{
                            const p = params[0];
                            return `${{p.name}}<br/>Δ S_Final: ${{p.value}}`;
                        }}
                    }},
                    grid: {{ left: 220, right: 30, bottom: 20, containLabel: true }},
                    xAxis: {{
                        type: 'value',
                        min: (val) => Math.floor(val.min - 1),
                        max: (val) => Math.ceil(val.max + 1)
                    }},
                    yAxis: {{
                        type: 'category',
                        data: axisModels,
                        axisLabel: {{
                            interval: 0,
                            formatter: (v) => truncateLabel(v, 26)
                        }}
                    }},
                    series: [{{
                        name: 'Δ S_Final',
                        type: 'bar',
                        data: axisValues,
                        label: {{ show: true, position: 'right' }},
                        itemStyle: {{
                            color: (params) => (params.value >= 0 ? '#2ecc71' : '#e74c3c')
                        }}
                    }}]
                }};
                arenaChart.setOption(option);
            }}

            baselineSelect.addEventListener('change', render);
            blockSelect.addEventListener('change', render);
            render();
        }}

        function initSlang() {{
            const el = document.getElementById('chart-slang');
            if (!el) return;
            slangChart = echarts.init(el);

            const stats = {{}};
            DB.slang.forEach(r => {{
                const m = r.model_id;
                if (!stats[m]) {{ stats[m] = {{ hit: 0, total: 0 }}; }}
                stats[m].hit += Number(r.hit || 0);
                stats[m].total += 1;
            }});

            const rows = Object.keys(stats).map(m => {{
                const rate = stats[m].total ? (stats[m].hit / stats[m].total) : 0;
                return {{ m, v: rate }};
            }}).sort((a,b) => b.v - a.v);

            const models = rows.map(r => r.m);
            const values = rows.map(r => Number((r.v * 100).toFixed(2)));

            const option = {{
                title: {{ text: 'Slang/Term Hit Rate (%)' }},
                tooltip: {{
                    trigger: 'axis',
                    axisPointer: {{ type: 'shadow' }},
                    formatter: (params) => {{
                        const p = params[0];
                        return `${{p.name}}<br/>Hit Rate: ${{p.value}}%`;
                    }}
                }},
                grid: {{ left: 220, right: 30, bottom: 20, containLabel: true }},
                xAxis: {{ type: 'value', max: 100 }},
                yAxis: {{
                    type: 'category',
                    data: models,
                    axisLabel: {{
                        interval: 0,
                        formatter: (v) => truncateLabel(v, 26)
                    }}
                }},
                series: [{{
                    name: 'Hit Rate (%)',
                    type: 'bar',
                    data: values,
                    label: {{ show: true, position: 'right' }},
                    itemStyle: {{ color: '#8e44ad' }}
                }}]
            }};
            slangChart.setOption(option);

            // Top terms by miss rate
            const termStats = {{}};
            DB.slang.forEach(r => {{
                const t = r.term || 'N/A';
                if (!termStats[t]) {{ termStats[t] = {{ hit: 0, total: 0 }}; }}
                termStats[t].hit += Number(r.hit || 0);
                termStats[t].total += 1;
            }});
            const termRows = Object.keys(termStats).map(t => {{
                const rate = termStats[t].total ? (termStats[t].hit / termStats[t].total) : 0;
                return {{ t, miss: 1 - rate }};
            }}).sort((a,b) => b.miss - a.miss).slice(0, 20);

            const termWrap = document.getElementById('slang-terms');
            if (termWrap) {{
                termWrap.innerHTML = termRows.map(r => `<span class="chip">${{r.t}}</span>`).join(' ');
            }}
        }}

        function buildDivergenceItems() {{
            const map = {{}};
            DB.divergence.forEach(d => {{
                const key = `${{d.test_id}}||${{d.model_id}}`;
                const type = d.type || (d.votes === 3 ? 'marginal_pass' : (d.votes === 2 ? 'marginal_fail' : 'other'));
                if (!map[key]) {{
                    map[key] = {{
                        test_id: d.test_id,
                        model_id: d.model_id,
                        block: d.block || 'N/A',
                        total: 0,
                        types: {{}},
                        severity: d.severity || d.consensus_severity || 'N/A'
                    }};
                }}
                map[key].types[type] = (map[key].types[type] || 0) + 1;
                map[key].total += 1;
            }});
            divergenceItems = Object.values(map).sort((a,b) => b.total - a.total);
        }}

        function renderDivergence(filterText = '') {{
            const divCont = document.getElementById('divergence-container');
            if (!divCont) return;

            if (!divergenceItems.length) {{
                divCont.innerHTML = '<p>No divergence cases found.</p>';
                return;
            }}

            const term = (filterText || '').toLowerCase();
            const filtered = divergenceItems.filter(d => {{
                const hay = `${{d.test_id}} ${{
                    d.model_id
                }} ${{
                    d.block
                }}`.toLowerCase();
                return hay.includes(term);
            }}).slice(0, 80);

            let html = '<ul class="list-group">';
            filtered.forEach(d => {{
                const types = Object.entries(d.types).map(([k,v]) => `${{k}}×${{v}}`).join(' | ');
                html += `<li class="list-group-item">
                    <strong>Test ID:</strong> ${{d.test_id}} <br>
                    <strong>Model:</strong> ${{d.model_id}} <br>
                    <strong>Block:</strong> ${{d.block}} <br>
                    <strong>Cases:</strong> ${{d.total}} <br>
                    <strong>Types:</strong> ${{types}} <br>
                    <em>Severity: ${{d.severity}}</em>
                </li>`;
            }});
            html += '</ul>';
            divCont.innerHTML = html;
        }}

        function setupResizeHandlers() {{
            document.querySelectorAll('button[data-bs-toggle="pill"]').forEach(btn => {{
                btn.addEventListener('shown.bs.tab', () => {{
                    if (leaderboardChart) leaderboardChart.resize();
                    if (arenaChart) arenaChart.resize();
                    if (slangChart) slangChart.resize();
                }});
            }});
            window.addEventListener('resize', () => {{
                if (leaderboardChart) leaderboardChart.resize();
                if (arenaChart) arenaChart.resize();
                if (slangChart) slangChart.resize();
            }});
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            initLeaderboard();
            initArena();
            initSlang();
            buildDivergenceItems();
            renderDivergence();
            setupResizeHandlers();

            const searchInput = document.getElementById('searchInput');
            if (searchInput) {{
                searchInput.addEventListener('input', function(e) {{
                    const term = e.target.value.toLowerCase();
                    document.querySelectorAll('.item-card').forEach(card => {{
                        const searchData = card.getAttribute('data-search').toLowerCase();
                        card.style.display = searchData.includes(term) ? 'block' : 'none';
                    }});
                }});
            }}

            const divSearch = document.getElementById('divergenceSearch');
            if (divSearch) {{
                divSearch.addEventListener('input', (e) => {{
                    renderDivergence(e.target.value || '');
                }});
            }}

            const audit = document.getElementById('audit-json');
            if (audit) {{
                audit.textContent = JSON.stringify(DB.judge_consensus, null, 2);
            }}
        }});
    </script>
</body>
</html>
"""

    out_html.write_text(html_content, encoding="utf-8")
    print(f"Interactive Wall V2 successfully generated at {out_html}")


if __name__ == "__main__":
    build_interactive_wall()
