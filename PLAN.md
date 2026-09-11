# Exp110 COBA gradient-damping alignment

Wall-time estimates assume A100-class GPUs and exclude scheduler queue time.

1. [x] Confirm the scope is exactly all 21 COBA cells: three full-MNIST `TR-01` canonical cells plus eighteen `TR-02` activity-frontier cells. **Estimate: 10 min.**
2. [x] Set all 21 COBA cells to `v_grad_dampen = 1000` while retaining `ei_strength = 0`; do not change PING or other training families. **Estimate: 30–60 min.**
3. [x] Update exp022 configuration and scientific-contract validation to require damping 1000 for every COBA cell. **Estimate: 30–60 min.**
4. [x] Add a provenance-safe replacement-bank path that reuses 81 unchanged cells and retrains 21 cells while producing one complete 102-cell bank. **Estimate: 1–2 h.**
5. [x] Verify that reused cells retain exact payload hashes and source references and that replacement cells retain both `weights.pth` and `weights_final.pth`. **Estimate: 30 min.**
6. [x] Commit and push the execution changes before running tests or preparing HPC work. **Estimate: 10–20 min.**
7. [x] Run focused exp022 contract and bank tests after the commit and push. **Estimate: 10–20 min.**
8. [x] Prepare and review the frozen exp022 HPC plan and perform scheduler test-only checks for its compute and collector links. **Estimate: 20–30 min plus queue response.**
9. [ ] Run the exp022 replacement compute bank with 21 concurrent GPUs. **Estimate: 9.5–12 h; approximately 51 GPU-hours.**
10. [ ] Validate the completed 102-cell exp022 bank, including `v_grad_dampen = 1000` and `ei_strength = 0` for all 21 replacement cells and byte-identical reuse for the other 81. **Estimate: 10–20 min.**
11. [ ] Run exp025 compute from the replacement exp022 bank. **Estimate: 1 h 20 min–1 h 40 min on one GPU.**
12. [ ] Run exp025 analyse from the new exp025 compute run. **Estimate: <1 min.**
13. [ ] Run exp025 present from the new exp025 analysis run, producing the replacement Figure 3 assets and `numbers.json`. **Estimate: <1 min.**
14. [ ] Run exp038 compute from the replacement exp022 bank. **Estimate: 45–60 min on one GPU.**
15. [ ] Run exp038 analyse from the new exp038 compute run. **Estimate: <1 min.**
16. [ ] Run exp038 present from the new exp038 analysis run, producing the replacement Figure 4 assets and `numbers.json`. **Estimate: <1 min.**
17. [ ] Run exp037 compute from the replacement exp022 bank using the six-shard production path. **Estimate: 45–55 min on six concurrent GPUs.**
18. [ ] Run exp037 analyse from the new exp037 compute run. **Estimate: <1 min.**
19. [ ] Run exp037 present from the new exp037 analysis run, producing the replacement Figure 7 assets and `numbers.json`. **Estimate: <1 min.**
20. [ ] Compare unchanged PING measurements with the previous results to detect accidental recipe or evaluation drift. **Estimate: 20–40 min.**
21. [ ] Update exp110's pinned exp037 presentation source while retaining the accepted exp054, exp041, exp046 and exp044 sources. **Estimate: 10 min.**
22. [ ] Run exp110 present to create a synthesis lineage containing the replacement exp037 evidence. **Estimate: 1–2 min.**
23. [ ] Select the new exp025, exp037, exp038 and exp110 presentation runs for the manuscript. **Estimate: 10 min.**
24. [ ] Recalculate and revise every affected number and interpretation in exp110 Results, captions, Methods, parameter tables and appendices. **Estimate: 1–3 h.**
25. [ ] Remove the unequal-damping manuscript note only after the new lineage and wording are verified. **Estimate: 5 min.**
26. [ ] Rebuild exp110 and check input coverage, ancestry, source links, equations, figure labels, rendered HTML and numerical consistency. **Estimate: 20–40 min.**
27. [ ] Record the replacement rationale and final run IDs without altering any historical run. **Estimate: 10–20 min.**
