import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.2)
ax.axis("off")

def box(x, y, w, h, text, color="#EAF2FB", edge="#2C5F8A", fontsize=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                        linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            wrap=True)

def arrow(x1, y1, x2, y2, color="#444444"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.2, color=color)
    ax.add_patch(a)

# Layer labels
ax.text(0.95, 5.9, "PERCEPTION", fontsize=10, fontweight="bold", color="#2C5F8A")
ax.text(4.55, 5.9, "PREDICTION", fontsize=10, fontweight="bold", color="#2C5F8A")
ax.text(6.35, 5.9, "DECISION", fontsize=10, fontweight="bold", color="#2C5F8A")
ax.text(8.85, 5.9, "EXPLANATION", fontsize=10, fontweight="bold", color="#2C5F8A")

# Perception layer
box(0.4, 4.4, 1.9, 1.1, "Sensor Agents\n(x S)\nLSTM-AE per sensor\nEq. 2-5", fontsize=8)
box(0.4, 2.7, 1.9, 1.1, "Master Sensor\nAggregator Agent\nfractions + flags")
box(2.7, 2.9, 1.9, 1.6, "Adaptive Window\nAgent\nGDRNet MLP + NSP\nWDI, FDI, WSS, FDS\nEq. 6-12", fontsize=8)

# Prediction layer
box(5.0, 3.6, 1.9, 1.6, "Prediction Agent\n(Transformer,\nfrozen baseline)\n[P_normal,P_warn,\nP_fault]  Eq. 13", fontsize=8)

# Decision layer
box(6.6, 1.6, 2.2, 2.0,
    "DECISION AGENT\nDeterministic policy\nMargin gating,\nnear-miss recovery\nAlgorithms 1 & 2\nTable 3", fontsize=8.5,
    color="#FCEFC7", edge="#B8860B")

# Explanation layer
box(9.1, 1.8, 1.6, 1.6, "Expert Agent\n(LLM, non-\nintervening)\nEq. 19", fontsize=8)
box(9.0, 0.2, 1.8, 1.0, "Human / Maintenance\nOperator", fontsize=8)

# arrows perception -> aggregator
arrow(1.35, 4.4, 1.35, 3.8)
arrow(2.3, 3.5, 2.7, 3.6)          # sensor agents -> window agent (shared data)
arrow(1.35, 2.7, 6.6, 2.5)         # aggregator -> decision agent
arrow(3.65, 2.9, 6.6, 2.5)         # window agent -> decision agent (drift/context)
arrow(5.95, 4.3, 6.6, 3.2)         # prediction -> decision agent

# decision -> expert (conditional) and decision -> alert/output
arrow(8.8, 2.6, 9.1, 2.6)
arrow(8.7, 1.8, 9.9, 1.2)
ax.text(6.7, 0.9, "Fault Detection, Fault Warning, Alert\n(deterministic, auditable)",
        fontsize=8, style="italic", color="#555555")
arrow(7.7, 1.6, 7.7, 1.15)

ax.text(0.4, 0.35,
        "Figure: ASPIRE architecture (reproduced schematic based on Guha & Datta, 2026, Fig. 1).\n"
        "Learning-based agents provide EVIDENCE only; alerts are committed solely by the deterministic Decision Agent.",
        fontsize=7.5, color="#333333")

plt.tight_layout()
plt.savefig("docs/architecture_diagram.png", dpi=200, bbox_inches="tight")
print("saved docs/architecture_diagram.png")
