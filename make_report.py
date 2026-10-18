"""Rebuild the three-page report from saved results; no model/GPU execution."""
from pathlib import Path
import argparse
import json
import textwrap
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Assignment1_results'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def paragraph(fig, y, text, size=10.2):
    wrapped = '\n'.join(textwrap.fill(p, width=103) for p in text.split('\n'))
    fig.text(.065, y, wrapped, fontsize=size, va='top', linespacing=1.35)


def heading(fig, y, text):
    fig.text(.065, y, text, fontsize=13, weight='bold', va='top', color='#183a57')


def page(number):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.065, .975, 'Big Data Analytics | Assignment 1 | PiSSA', fontsize=9, color='#52616c', va='top')
    fig.text(.935, .025, str(number), ha='right', fontsize=9)
    return fig


def bars(ax, vals, labels, ylim, title):
    rects = ax.bar(labels, vals, color=['#277da1', '#43aa8b', '#f9c74f', '#f9844a'][:len(vals)], width=.6)
    ax.bar_label(rects, labels=[f'{v:.1f}' for v in vals], padding=3, fontsize=10)
    ax.set(ylabel='Accuracy (%)', ylim=(0, ylim), title=title)
    ax.tick_params(axis='x', labelsize=9)


def q4_pair(fig, layer, bottom):
    import numpy as np
    with np.load(OUT / 'q4_spectra.npz') as spectra:
        values = spectra[f'layer_{layer}'].astype(float)
    energy = np.cumsum(values**2) / np.sum(values**2) * 100
    rank = np.searchsorted(energy, 50) + 1
    for i, kind in enumerate(('singular_values', 'cumulative_energy')):
        ax = fig.add_axes([.10 + i*.46, bottom+.045, .36, .17])
        x = np.arange(1, 201)
        if kind == 'singular_values':
            ax.bar(x, values[:200], width=.9, color='#277da1')
            ax.set(title=f'Layer {layer}: singular values', xlabel='Singular value index', ylabel='Magnitude')
        else:
            ax.plot(x, energy[:200], color='#277da1', linewidth=1.5)
            ax.axhline(50, color='#e76f51', ls='--', linewidth=1)
            ax.axvline(rank, color='gray', ls=':', linewidth=1)
            ax.text(.97, .08, f'50% at rank {rank}', transform=ax.transAxes, ha='right', fontsize=9)
            ax.set(title=f'Layer {layer}: cumulative energy', xlabel='Target rank', ylabel='Energy (%)', ylim=(0,100))
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id', default='[ADD STUDENT ID]')
    parser.add_argument('--github-url', default='https://github.com/chester71206/big_data_HW1')
    args = parser.parse_args()
    df = pd.read_csv(OUT / 'benchmark_results.csv')
    assert len(df) == 7 and df.status.eq('ok').all()
    ranks = df[df.component.eq('default')].sort_values('rank')
    base = df[df.method.eq('fresh')].iloc[0]
    components = ['default', '25pct', '50pct', 'bottom']
    labels = ['Top 16', '25%', '50%', 'Bottom 16']
    q3 = df[df['rank'].eq(16)].set_index('component').loc[components]
    energy = pd.read_csv(OUT / 'q4_minimum_ranks.csv').set_index('layer')
    assert energy.loc[[0,7,15], 'minimum_rank_for_50pct_energy'].tolist() == [62,120,84]

    # Audit saved predictions without changing the evaluation or inventing stop reasons.
    audit = []
    for row in df.itertuples():
        records = [json.loads(line) for line in (OUT / f'{row.run_tag}_predictions.jsonl').open()]
        records = [r for r in records if r['task'] == 'math']
        missing = [r for r in records if r['pred_answer'] is None]
        assert len(records) == row.math_total and len(missing) == row.math_unparsed
        assert sum(r['correct'] for r in records) == row.math_correct
        audit.append({'run_tag': row.run_tag, 'math_total': len(records), 'math_unparsed': len(missing),
                      'stop_reason_available': False, 'generated_token_count_available': False})
    pd.DataFrame(audit).to_csv(OUT / 'math_prediction_audit.csv', index=False)

    # Save Q3 plots in the requested singular-value order.
    for task, title, limit in [('gsm8k', 'GSM8K', 20), ('math', 'MATH', 5)]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars(ax, q3[f'{task}_accuracy'].values*100, labels, limit, f'{title}: rank 16')
        ax.set_xlabel('Selected singular-value range')
        fig.tight_layout()
        fig.savefig(OUT / 'q3_figures' / f'{task}_accuracy_by_component.png', dpi=200)
        plt.close(fig)

    with PdfPages(OUT / 'Asg1_DRAFT.pdf') as pdf:
        fig = page(1)
        heading(fig, .94, 'Rank, accuracy, and training cost')
        fig.text(.065, .91, f'Student ID: {args.student_id}', fontsize=10)
        fig.text(.065, .89, f'Code: {args.github_url}', fontsize=9)
        paragraph(fig, .863, 'Setup: Llama-3.2-1B; 24,975 MetaMathQA training examples; one epoch; learning rate 2e-5; FP16; effective batch 128; seed 42; one RTX 3090. Each task uses 1,000 evaluation examples and at most 256 generated tokens.')
        heading(fig, .792, 'Q1. Accuracy versus target rank')
        ax = fig.add_axes([.10, .565, .84, .183])
        import numpy as np
        x = np.arange(4)
        for offset, task, title, color in [(-.18, 'gsm8k', 'GSM8K', '#277da1'), (.18, 'math', 'MATH', '#f9844a')]:
            vals = [base[f'{task}_accuracy']*100] + (ranks[f'{task}_accuracy']*100).tolist()
            rects = ax.bar(x+offset, vals, width=.34, label=title, color=color)
            ax.bar_label(rects, labels=[f'{v:.1f}' for v in vals], padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=['Original LLM\n(Baseline)', 'Rank 16', 'Rank 64', 'Rank 256'], ylabel='Accuracy (%)', ylim=(0,33))
        ax.legend(loc='upper left', frameon=False)
        paragraph(fig, .51, 'Fine-tuning improves both tasks over the original LLM. From rank 16 to 256, GSM8K rises from 15.6% to 27.3%, and MATH from 3.7% to 5.7%. Trainable parameters grow from 11.27M to 45.09M to 180.36M. The larger adaptation subspace may help learn mathematical reasoning patterns, at greater memory and compute cost. This single-seed comparison supports a trend here, not a guarantee for other settings.')
        heading(fig, .405, 'Q2. Training time versus target rank')
        ax = fig.add_axes([.10, .205, .84, .16])
        times = ranks.train_minutes.tolist()
        rects = ax.bar(['Rank 16', 'Rank 64', 'Rank 256'], times, color='#43aa8b', width=.5)
        ax.bar_label(rects, labels=[f'{v:.2f}' for v in times], padding=3)
        ax.set(ylabel='Training time (min)', ylim=(0,35))
        paragraph(fig, .167, 'Time is not monotonic: rank 64 is 1.27 min faster than rank 16, while rank 256 is slowest. Runtime variability and GPU execution efficiency may explain the small reversal; one run per setting cannot isolate the cause. Rank 256 adds adapter computation and optimizer work. The shared base-model computation also limits proportional scaling with rank. Timing covers the training loop only, excluding SVD initialization, merging, and evaluation.')
        pdf.savefig(fig); plt.close(fig)

        fig = page(2)
        heading(fig, .94, 'Q3. Which singular-value range should be adapted?')
        for i, (task, title, limit) in enumerate([('gsm8k', 'GSM8K', 20), ('math', 'MATH', 5)]):
            ax = fig.add_axes([.10+i*.46, .655, .36, .22])
            bars(ax, q3[f'{task}_accuracy'].values*100, labels, limit, title)
        paragraph(fig, .603, 'All four runs use rank 16. Top 16 performs best on both tasks (15.6% GSM8K, 3.7% MATH). Moving to 25%, 50%, and the bottom generally reduces accuracy; MATH ties at 2.4% for 50% and Bottom 16. The 25% and 50% settings select 16 consecutive components starting near those positions in the ordered spectrum.')
        paragraph(fig, .515, 'In this experimental setting, adapting the leading singular directions appears more effective. These directions carry more weight-matrix energy, which may provide a useful starting subspace for adaptation. Energy alone does not establish task relevance, and a single seed cannot establish whether small gaps (such as 11.8% versus 11.7%) are reliable.')
        heading(fig, .414, 'Q4. Original q_proj spectra and retained energy')
        paragraph(fig, .381, 'Full SVD is computed for the original model. Energy at rank r is sum(s_i^2, i <= r) / sum(s_i^2, all i). Both plots show the first 200 components, but the denominator uses the entire spectrum. The minimum ranks retaining at least 50% energy are:')
        ax = fig.add_axes([.16, .262, .68, .055]); ax.axis('off')
        table = ax.table(cellText=[['Layer 0', 'Layer 7', 'Layer 15'], ['62', '120', '84']], cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1,1.35)
        q4_pair(fig, 0, .035)
        pdf.savefig(fig); plt.close(fig)

        fig = page(3)
        heading(fig, .94, 'Q4. Layer comparison (continued)')
        q4_pair(fig, 7, .665)
        q4_pair(fig, 15, .412)
        heading(fig, .412, 'Observations and interpretation')
        paragraph(fig, .38, 'Layer 0 has a pronounced leading singular value and a sharp early drop. Its cumulative energy reaches 50% at rank 62, showing the strongest concentration among these layers. Layer 7 has a flatter spectrum after its leading values and reaches 50% later, at rank 120. Layer 15 is intermediate, crossing at rank 84. Thus, the bar plots and cumulative curves agree with the minimum-rank ordering.')
        paragraph(fig, .28, 'At the 50% energy target, layer 0 admits a lower-rank approximation than layers 15 and 7. The squared singular values measure retained Frobenius-norm energy, so different layers have different weight-approximation characteristics. This does not identify which layer matters most for model accuracy or determine the best fine-tuning rank; no layer-wise ablation was performed.')
        heading(fig, .175, 'Evaluation limitations')
        paragraph(fig, .145, 'MATH predictions without an extracted answer: baseline 958/1,000; Top-rank 16/64/256: 552/470/452; rank-16 25%/50%/Bottom: 584/602/599. Thus, the fine-tuned range is 452-602. Some saved texts visibly end mid-expression, but token counts and stop reasons were not logged; truncation versus unrecognized answer format cannot be counted reliably. Unparsed answers count as incorrect. Results reflect generation and extraction, with one seed and 1,000 examples per task.', size=9.7)
        pdf.savefig(fig); plt.close(fig)
    print(OUT / 'Asg1_DRAFT.pdf')


if __name__ == '__main__':
    main()
