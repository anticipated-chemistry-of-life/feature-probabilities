# Retain only the top-k (k = 200) structure candidates per Feature

SIRIUS's expansive structure search falls back to PubChem when it finds no confident structure candidate in BIO. Before this decision, chunks that hit the fallback were huge: MassSpecGym chunk orbitrap_8 produced 11.2 M Annotations for 10 000 Features (mean 1 122, max rank 57 368), while chunks without it averaged 120–290 per Feature. orbitrap_8 took ~69 h end to end; its SIRIUS job took 1.6 h and persisting its Annotations took the rest. So every SIRIUS run, field and ground truth alike, now keeps only each Feature's 200 best-ranked structure candidates (Top-k in `CONTEXT.md`), and k is part of the SIRIUS run's identity.

## Considered Options

- **Have SIRIUS compute fewer candidates.** Rejected: SIRIUS 6.5.4 offers no working control. The config key `NumberOfStructureCandidates` (passable through `JobSubmission.configMap`) is accepted but has no effect on the stored candidates (k = 5 still stored all 31). Reducing formula candidates did not reduce structure candidates either. So k is applied when the results are read back, as the first page of `get_structure_candidates_page`. SIRIUS still does the full search; the savings are in transfer, persistence and storage.
- **Narrow the search space instead** (turn off the expansive PubChem fallback). Not chosen: it changes which structures can be found at all, rather than how many are kept.
- **Apply k to ground truth runs only.** Rejected: the calibration score is fitted on ground truth runs and applied to field runs, so both must see the same candidate population.
- **Leave k out of the caching key.** Rejected: a cached run read under a different k would be silently reused. k sits in the analysis-params checksum, the same way the feature-quality filter does.

## Consequences

- A ground truth Feature whose true structure ranks below k has no correct assignment and contributes nothing to the KDE fit. This shifts the fitted data toward higher CSI:FingerID scores. Accepted: candidates ranked below 200 are of no practical use for annotation.
- Changing k means recomputing every SIRIUS run. Runs made without k (the first generate-groundtruth attempt) were discarded by starting a fresh database, not migrated.
- SIRIUS's page sort parameter is inverted: `sort=rank,asc` returns the *worst* candidates first. Candidates are requested with `sort=csiScore,desc`, which is verified to return ranks 1..k.
