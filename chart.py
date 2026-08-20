import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('graphrag_rca_work/results/20260802_1234_results.csv')

GROUPS = ['Application logic', 'JVM runtime', 'Cloud resource',
          'Middleware&DB', 'K8s lifecycle', 'Resource&perf.']
GROUP_SHORT = ['App\nlogic', 'JVM\nruntime', 'Cloud\nresource',
               'Middleware\n&DB', 'K8s\nlifecycle', 'Resource\n&perf.']

systems = [s for s in ['direct_llm', 'standard_rag', 'graphrag_only',
                        'multi_agent_only', 'proposed_hybrid'] if s in df['system'].unique()]

system_labels = {
    'direct_llm': 'Direct LLM',
    'standard_rag': 'Standard RAG',
    'graphrag_only': 'GraphRAG Only',
    'multi_agent_only': 'Multi-Agent Only',
    'proposed_hybrid': 'Proposed Hybrid',
}

n = len(systems)
fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.5), dpi=200)
if n == 1:
    axes = [axes]

for ax, system in zip(axes, systems):
    sub = df[df['system'] == system]
    # build confusion matrix: rows = ground truth, cols = predicted
    cm = pd.crosstab(sub['gt_fault_group'], sub['predicted_fault_group'])
    cm = cm.reindex(index=GROUPS, columns=GROUPS, fill_value=0)

    im = ax.imshow(cm.values, cmap='Blues', vmin=0)
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            v = cm.values[i, j]
            is_diag = (i == j)
            color = 'white' if v > cm.values.max() * 0.5 else 'black'
            weight = 'bold' if is_diag else 'normal'
            ax.text(j, i, str(v), ha='center', va='center', color=color,
                     fontsize=10, fontweight=weight)
            if is_diag:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                             edgecolor='#d62728', linewidth=2))

    ax.set_xticks(range(len(GROUPS)))
    ax.set_yticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUP_SHORT, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(GROUP_SHORT, fontsize=8)
    ax.set_xlabel('Predicted group', fontsize=9)
    ax.set_ylabel('Ground-truth group', fontsize=9)
    acc = (sub['gt_fault_group'] == sub['predicted_fault_group']).mean()
    ax.set_title(f"{system_labels.get(system, system)}\n(fault_group accuracy = {acc:.3f})",
                 fontsize=10, fontweight='bold')

fig.suptitle('Fault-Group Confusion Matrix (6x6)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphrag_rca_work/outputs/chart_confusion_matrix_fault_group.png', dpi=200, bbox_inches='tight')
print('Saved chart_confusion_matrix_fault_group.png')