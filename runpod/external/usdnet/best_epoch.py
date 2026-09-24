"""Best-epoch summary over per-epoch validation logs (USDNet CSVLogger metrics.csv).
For each arm: final epoch, best epoch by each column, and a cross-selection proxy
(best epoch chosen on M_ap50 → report all columns at that epoch; chosen on MAO_ap50 →
report the others). Their evaluator logs only aggregates per pass, so a scene-level
split-half selection is not available without modifying their eval."""
import sys, glob, pandas as pd
cols = ['M_ap50', 'MA_ap50', 'MO_ap50', 'MAO_ap50', 'MAO_ST_ap50']
for arm in sys.argv[1:]:
    f = glob.glob(f'/workspace/runs/ft_{arm}/**/metrics.csv', recursive=True)[0]
    d = pd.read_csv(f)[['epoch'] + cols].dropna(subset=['M_ap50']).reset_index(drop=True)
    d['epoch'] = d['epoch'].astype(int) + 1
    print(f'== {arm}  ({len(d)} validations)')
    print(d.round(3).to_string(index=False))
    last = d.iloc[-1]; print('final   :', ' '.join(f'{last[c]:.3f}' for c in cols))
    for sel in ['M_ap50', 'MAO_ap50', 'MA_ap50']:
        i = d[sel].idxmax(); r = d.loc[i]
        print(f'best-by-{sel:9s} ep{int(r.epoch):2d}:', ' '.join(f'{r[c]:.3f}' for c in cols))
    print('mean over epochs>=10:', ' '.join(f'{d[d.epoch>=10][c].mean():.3f}' for c in cols))
