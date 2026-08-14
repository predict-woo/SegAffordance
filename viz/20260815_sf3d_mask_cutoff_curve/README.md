# 20260815_sf3d_mask_cutoff_curve

Records available to train on vs `min_mask_area_frac` cutoff, measured
over the FULL database (458k records) after the standing sensor
(occluded_frac 0.5) and radius (min_revolute_radius 0.10) filters —
i.e. the same base split gen-6/7 train on (356,747 records, count
reproduced exactly by this scan).

Key points (splat pixels / W*H, original resolution):

| cutoff | records | share |
|---|---|---|
| none        | 356,747 | 100%  |
| 0.01%       | 293,291 | 82.2% |
| 0.05%       | 120,182 | 33.7% |
| 0.10%       |  59,615 | 16.7% |
| 0.25% (g8)  |  19,296 |  5.4% |

Distribution: p50 = 0.030%, p75 = 0.068%, p90 = 0.157% — the knee of
the curve sits right on top of the useful cutoff range, so data cost
rises steeply between 0.01% and 0.25%. E.g. relaxing gen-8's 0.25% to
0.10% triples the data (19.3k -> 59.6k); 0.05% is a 6x increase while
still dropping the smallest third of elements.

Regenerate: tools-free one-off — scan script + plot script recorded in
the session (scan: unpickle all records, apply the two standing
filters, save fracs; plot: survivors = total - searchsorted(sorted
fracs, cutoffs)). Fracs array: /tmp/mask_fracs_postfilter.npy on the
dev pod (disposable).
