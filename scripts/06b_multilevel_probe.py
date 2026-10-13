"""Do multilevel merge-split moves cross between basins? (Section 6.1)

Starts from the last sample of the highest-DL (seed 101) and lowest-DL (seed 104) beta=1
multiflip chains of 06_posterior_sample.py and runs 40 calls of
multilevel_mcmc_sweep(beta=1, niter=1), printing the DL after every call.

Usage:  python scripts/06b_multilevel_probe.py > results/posterior/calm_2023_24/multilevel_probe.log
"""
import os, sys, gzip, pickle, time, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import graph_tool.all as gt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("s06", f"{ROOT}/scripts/06_posterior_sample.py")
s06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s06)

def probe(idx, n_calls=40):
    run = pickle.load(gzip.open(f"{ROOT}/results/posterior/calm_2023_24/chains.pkl.gz", "rb"))
    g, _ = s06.window_graph("calm_2023_24")
    bs = [np.array(b) for b in run["chains"][idx]["samples"][-1]]
    st = gt.NestedBlockState(g, bs=bs, base_state=gt.WeightedBlockState,
                             base_state_args=dict(deg_corr=False, rec=[g.ep["w"]], rec_types=["real-normal"]))
    gt.seed_rng(7 + idx)
    seed = run["chains"][idx]["seed"]
    t0 = time.time()
    print(f"chain {seed}: start DL {st.entropy():.1f}", flush=True)
    for i in range(n_calls):
        st.multilevel_mcmc_sweep(beta=1, niter=1)
        print(f"chain {seed} call {i+1}: DL {st.entropy():.1f}  B {len(np.unique(st.get_bs()[0]))}  "
              f"t {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        list(ex.map(probe, [0, 3]))
