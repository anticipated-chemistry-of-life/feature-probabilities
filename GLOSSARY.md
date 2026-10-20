# Feature Probabilities

Turns an mzML file into a table of probabilities that a given molecule is the correct identity of a detected LC-MS feature, calibrated against MSnLib ground truth via SIRIUS/CSI:FingerID.

## Language

**Feature**:
One MS2 spectrum treated as an analytical unit by SIRIUS — either detected via SIRIUS's peak-picking from an mzML run, or supplied already-picked (e.g. one MSnLib spectrum, imported without peak-picking). The unit being annotated.
_Avoid_: peak, spectrum (ambiguous with a raw, unpicked scan)

**Structure candidate**:
A molecule SIRIUS/CSI:FingerID proposes as a possible identity for a feature, carrying a CSI:FingerID score. Returned by `get_structure_candidates` (PySirius `StructureCandidateFormula`) — this row does NOT itself carry ion mass. Per-feature: the same underlying Molecule can appear as a structure candidate on many features.
_Avoid_: molecule (see the distinct **Molecule** entry below — the two are not interchangeable), hit

**Molecule**:
The globally deduplicated, skeleton-level chemical identity a structure candidate resolves to: a first-block (14-character) InChIKey plus a canonical, stereochemistry-stripped 2D SMILES (or SIRIUS's SMILES verbatim when RDKit can't parse it — SIRIUS's candidate databases contain valence-violating structures). Stored once regardless of how many features/structure candidates reference it. SIRIUS's candidate search already operates at this granularity, and the ground truth set's full InChIKeys are cut to their first block before comparison, so a Molecule never distinguishes stereoisomers of the same skeleton.
_Avoid_: structure candidate (that's the per-feature, scored occurrence; a Molecule is the shared, unscored identity)

**Annotation**:
One feature–structure-candidate pair: a structure candidate's CSI:FingerID score joined to its feature's ion mass. Ion mass lives on the feature (PySirius `AlignedFeature.ionMass`), not on the structure-candidate row, so producing an Annotation requires joining the two by feature id — the row unit consumed by calibration.

**Extract**:
The physical sample (a species plus, e.g., organ) that one or more SIRIUS runs are acquired from. Purely biological/sample provenance — acquisition-specific detail (ionization mode, instrument type) lives on the SIRIUS run, not here.

**SIRIUS run**:
One execution of SIRIUS against one input spectra file (an mzML or an MGF chunk) with one fixed parameter set, identified by the rerun-avoidance caching key. A **field run** processes a real mzML tied to an Extract; a **ground truth run** processes a chunk of the ground truth set and has no Extract. Every SIRIUS run has exactly one Ionization mode. Reprocessing the same input file with different parameters produces a new, coexisting SIRIUS run — never overwrites a prior one.
_Avoid_: job (SIRIUS's own API term for the async task that executes a run — a run is the durable record, a job is how it got produced)

**Top-k**:
The maximum number of best-ranked (by CSI:FingerID score) structure candidates per Feature that a SIRIUS run retains; lower-ranked structure candidates never become Annotations. Part of a SIRIUS run's parameter set, so runs with different k are different SIRIUS runs. Applies to field runs and ground truth runs alike, so the calibration score is only ever fitted on and applied to the same candidate population. A ground truth Feature whose true structure ranks below k has no correct assignment.
_Avoid_: candidate cap, candidate limit, max candidates

**Ground truth set**:
The `ALL_ENERGIES` spectra of MSnLib (each a merge of one compound's MS2 scans across all collision energies) at a pinned dataset revision, in both Ionization modes, all acquired on Orbitrap. Excludes spectra whose identity is ambiguous (matched to more than one library compound) and spectra whose adduct SIRIUS cannot detect. Used to fit the calibration score's KDEs.
_Avoid_: MassSpecGym (the former ground truth set)

**Ionization mode**:
The charge polarity, positive or negative, a spectrum was acquired in. A property of the SIRIUS run, not of the Extract: one mzML is acquired in one Ionization mode, and the same Extract may be run in both. Calibration is split by Ionization mode and by nothing else.
_Avoid_: polarity, ion mode, ionisation mode

**Instrument type**:
The mass-analyser class a spectrum was acquired on, Orbitrap or QTOF: the only two classes SIRIUS has an instrument profile for. A property of the SIRIUS run, where it selects that run's SIRIUS instrument profile; it does not split calibration. A field run's instrument type is resolved from the instrument model its metadata row names.
_Avoid_: instrument model (the specific product, e.g. "LC-MS QExactive Plus", which resolves to an instrument type), instrument profile (SIRIUS's settings preset an instrument type selects)

**Correct assignment**:
A structure candidate whose InChIKey first block (skeleton/connectivity layer only) matches the ground truth set's known true structure for that spectrum. Stereochemistry and protonation state are ignored, since MS/MS routinely can't resolve them.
_Avoid_: exact match, true positive (without qualifying "skeleton-level")

**Calibration score**:
The value returned for a feature–candidate pair's (ion mass, CSI:FingerID score) by the KDE fitted on the ground truth set's correct assignments in that pair's Ionization mode. One KDE per Ionization mode; instrument type does not split calibration. A density used to rank candidates within a feature, not a normalized posterior probability.
_Avoid_: probability, confidence (both imply a calibrated posterior, which this isn't)

**Feature-probability table**:
`annotate`'s final exported table (one row per feature–structure-candidate pair with its calibration score) — the deliverable the whole pipeline produces for a batch of new mzML files.
