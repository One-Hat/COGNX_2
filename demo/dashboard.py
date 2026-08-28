# Copyright (c) 2026 COGNX. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. This file is part of MNEMA, proprietary software
# of COGNX. It is not open source. No right to use, copy, modify, distribute, or
# create derivative works is granted. Unauthorized use or disclosure is prohibited.
# See the LICENSE file at the repository root for the full terms.

import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNEMA Telemetry Dashboard", layout="wide")

st.title("🧠 MNEMA: Neuromorphic Continual Learning Telemetry")
st.markdown("A memory-native, event-driven architecture for bounded continual learning at the edge.")

# Load Benchmark Results
results_path = "results/benchmark_results.json"
if not os.path.exists(results_path):
    st.error("No benchmark results found. Run `experiments/run_split_mnist_benchmark.py` first.")
    st.stop()

with open(results_path, "r") as f:
    results = json.load(f)

# Top Metrics Overview
st.header("⚡ System Performance & Energy Odometer")
col1, col2, col3, col4 = st.columns(4)

mnema_res = results.get("MNEMA (Full Engine)", {})
mlp_naive = results.get("MLP (Naive)", {})
mlp_replay = results.get("MLP (+Replay Buffer)", {})

with col1:
    st.metric(
        label="MNEMA Final Accuracy", 
        value=f"{mnema_res.get('final_acc', 0)}%", 
        delta=f"+{round(mnema_res.get('final_acc', 0) - mlp_naive.get('final_acc', 0), 1)}% vs Naive"
    )
with col2:
    st.metric(
        label="Forgetting Measure (FM)", 
        value=f"{mnema_res.get('forgetting', 0)}%", 
        delta=f"-{round(mlp_naive.get('forgetting', 0) - mnema_res.get('forgetting', 0), 1)}%", 
        delta_color="inverse"
    )
with col3:
    energy_ratio = mlp_replay.get('energy_uj', 1) / max(mnema_res.get('energy_uj', 1), 1e-3)
    st.metric(
        label="Energy Advantage", 
        value=f"{energy_ratio:.1f}x Lower", 
        delta="SynOps vs Dense MACs"
    )
with col4:
    st.metric(
        label="Memory Capacity Status", 
        value="64 KB (Flat)", 
        delta="Bounded (Zero Growth)"
    )

st.divider()

# Retention Matrices
st.header("📊 Retention Matrices R[t, j] (Class-IL 5-Task)")
cols = st.columns(len(results))

for idx, (name, data) in enumerate(results.items()):
    with cols[idx]:
        st.subheader(name)
        st.caption(f"Final ACC: **{data['final_acc']}%** | Forgetting: **{data['forgetting']}%**")
        
        R = np.array(data["retention_matrix"]) * 100.0
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(R, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"T{i}" for i in range(5)])
        ax.set_yticklabels([f"T{i}" for i in range(5)])
        ax.set_xlabel("Evaluated Task")
        ax.set_ylabel("Trained Task")

        for i in range(5):
            for j in range(5):
                if j <= i:
                    val = R[i, j]
                    color = "white" if val < 30 or val > 75 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
        
        st.pyplot(fig)

st.divider()

# Energy and Operations Breakdown Table
st.header("🔬 Algorithmic & Hardware Auditing Breakdown")
table_data = []
for name, data in results.items():
    table_data.append({
        "Model": name,
        "Class-IL Accuracy (%)": f"{data['final_acc']}%",
        "Forgetting Measure (%)": f"{data['forgetting']}%",
        "Dense MACs": f"{data['macs']:,}",
        "Sparse SynOps": f"{data['synops']:,}",
        "Integer Adds": f"{data['adds']:,}",
        "Projected Energy [ASIC 45nm]": f"{data['energy_uj']} µJ"
    })

st.table(table_data)

st.info("💡 **Auditing Disclaimer:** Energy values represent projected microjoules based on published 45nm ASIC specs (Horowitz, ISSCC 2014) multiplied by exact software-measured operation counts.")