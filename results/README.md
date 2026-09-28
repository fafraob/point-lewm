# Results

Logged outcomes behind the paper's tables, for checking and for the statistical tests.

| File | Content |
|---|---|
| `table2_per_seed.csv` | Success rate of every Table 2 cell per evaluation seed (10 seeds x 50 episodes). The image Delta-JEPA row is absent from the CSV; its sweep is `config/eval_sweep/image_<env>_delta_jepa.conf` on the released `checkpoints/<env>/image-delta-jepa/` checkpoint. Columns: table, environment, family, observation, method, seed, num_episodes, success_rate. `scripts/paired_stats.py` computes the tests of Table 12 from it. |
| `probing/probe_<env>.json` | Linear and MLP probe results of Tables 6-8 (test MSE and R^2 per target, the ridge regularizer and the MLP's best epoch), as written by `probe_latents.py`. |
| `probing/probe_report.txt` | The same results as the text report of `probe_latents.py --report`. |

The probing results were computed with the released Point-LeWM and Point-Delta-JEPA checkpoints
of each environment (`checkpoints/<env>/point-lewm/`, `checkpoints/<env>/point-delta-jepa/`, the ones
the planning results of Table 2 use); the run labels in the files (`point_lewm_<env>`,
`point_deltajepa_<env>`) name those checkpoints.
