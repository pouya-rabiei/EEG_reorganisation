# EEG network reorganization

Baseline resting-state EEG analyses for transitioners, non-transitioners and healthy controls. Run the numbered Python scripts in order. Each statistics script saves CSV results; its following plot script reads those results without refitting models.

## Setup

Use a dedicated Python environment and install `requirements.txt`. These versions match the local validation environment (Python 3.14). The original scripts and manuscript results are not modified.

Set the project root and Schaefer centroid path in `config.py`, or supply these environment variables:

```powershell
$env:EEG_PROJECT_ROOT = 'D:/path/to/EEG_projects'
$env:SCHAEFER_CENTROIDS = 'D:/path/to/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
python 01_preprocessing.py
```

By default, group outputs go to `results/` beside the scripts. `EEG_RESULTS_DIR` can point to a different output folder. Individual preprocessing and source outputs use the project's existing `derivatives/PREPROCESS_CONN` and `derivatives/CONN_NETW` folders. `OVERWRITE=False` reuses completed individual outputs; group statistics and plots are regenerated when their scripts run. Run script 03 to collect the existing individual outputs before running group statistics.

## Run order

| Script | Purpose |
| --- | --- |
| `01_preprocessing.py` | Individual preprocessing and QC collection |
| `02_preprocessing_statistics.py` | Three-group preprocessing QC statistics; no group plots |
| `03_subject_features.py` | Source PSD, spectral parameters, connectivity, graph metrics and collection |
| `04_psd_statistics.py` | Age- and sex-adjusted PSD cluster inference |
| `05_psd_plots.py` | Group and difference PSD heatmaps |
| `06_fooof_statistics.py` | Network-level spectral-parameter GEE and parcel follow-up estimates |
| `07_fooof_plots.py` | Spectral-parameter boxplots and glass-brain maps |
| `08_whole_brain_topology_statistics.py` | Whole-brain clustering and efficiency GEE |
| `09_whole_brain_topology_plots.py` | Whole-brain distributions |
| `10_network_topology_statistics.py` | Within-network clustering and efficiency GEE |
| `11_network_topology_plots.py` | Topology boxplots and brain maps |
| `12_within_network_connectivity_statistics.py` | Within-network connectivity GEE |
| `13_within_network_plots.py` | Boxplots, parcel heatmaps and connectomes |
| `14_between_network_connectivity_statistics.py` | Between-network connectivity GEE |
| `15_between_network_plots.py` | Boxplots, heatmaps and circular plots |
| `16_nbs_statistics.py` | NBS at thresholds 3.0, 3.5 and 4.0 |
| `17_nbs_plots.py` | NBS heatmaps and connectomes from saved results |

`config.py` holds shared paths and settings. Each statistics script contains its own model fitting, contrasts, FDR correction and interaction gate; scripts 10, 12 and 14 do not depend on an external GEE helper. Scripts can be imported without starting analyses.

## Inputs

- BIDS EEG folders: `RAW/raw_data_ncbp` and `RAW/raw_data_ncbphc`.
- Demographics: `demo_ncbp.csv` and `demo_ncbpmc.csv` in those folders, with `subject`, `age` and `sex`; the patient table also needs `chroncity_class_6m`.
- Six-month outcome: `0 = NCBP_nontrans`, `1 = NCBP_trans`. The study-specific correction for `sub-329` is explicit in `load_subject_info()` in `config.py`.
- Source-group plus subject ID identifies a participant; outcome group does not replace that identifier. Participants need at least 50 retained epochs for the analytical sample.
- The Schaefer 100-parcel, seven-network centroid CSV must contain `ROI Name`, `R`, `A` and `S`. Matrix rows and columns are aligned by parcel name.
- MNE's fsaverage files and Nilearn's matching Schaefer atlas are downloaded/cached if needed. The source reconstruction uses one discrete source at each centroid, as in the current analysis code.

Script 03 saves the spectral tables, four graph/connectivity subject tables, participant metadata and a connectivity-file index. NBS reads the saved participant metadata and individual connectivity-matrix CSVs. No raw EEG or participant data are bundled here.

## Analysis conventions

- Planned contrasts: transitioners minus non-transitioners, transitioners minus controls, and non-transitioners minus controls. HC is the GEE reference group.
- PSD uses 1–45 Hz, age/sex-adjusted three-group linear models, HC3 standard errors and 5,000 restricted wild-bootstrap resamples. 
- FOOOF fits use 1–45 Hz. The source function's argument names now match that caller. Existing saved subject spectra/parameters are reused unless overwritten; their original fitting range is not changed by collection.
- Primary spectral-parameter inference remains network-level GEE. The existing parcel-level HC3 linear models are retained only for adjusted descriptive maps and standardized parcel follow-up estimates; they do not replace the GEE analysis.
- GEE includes age and categorical sex, participant clusters, Gaussian family, exchangeable working correlation and robust covariance. Original FDR families are preserved. `passes_interaction_gate` requires both the corrected interaction and corrected contrast to be below 0.05; `interaction_p_corrected` is also saved. Descriptive distributions are not restricted by this gate.
- Graph metrics use the original proportional threshold (20% density). Mean within- and between-network connectivity uses unthresholded connectivity weights. Parcel connectomes show the strongest 20% of absolute differences for display only.
- NBS uses **unadjusted pooled-variance Student t statistics**, not Welch statistics or age/sex-adjusted tests. Components are formed from `abs(t) > threshold`, so mixed-sign components are possible. Component extent is the number of unique edges. There are 5,000 label permutations, seed 42, shared across thresholds, with the `(b + 1)/(5000 + 1)` P-value correction. FWE control is within each comparison/measure/band/threshold analysis, not across all analyses. Individual edges do not receive corrected significance from NBS.
- Figures use concise titles identifying the group comparison, measure, band, metric and network as applicable. Glass-brain columns identify each group, effect size and significant effect size. Axes, colorbars, legends and the existing plotting style are retained. SVG export uses the existing 600-dpi save setting for rasterized elements. Blank significance maps mean no supported effect, not missing analysis.

## Validation

The scripts were run on the saved individual outputs for 29 transitioners, 43 non-transitioners and 65 controls. QC, PSD, all GEE analyses and all 72 NBS comparison/measure/band/threshold combinations were rerun. Statistical outputs matched the corrected subject-329 reference to numerical precision, with unchanged interaction-plus-contrast decisions. All seven plotting scripts completed. Raw EEG preprocessing and full source reconstruction were not rerun; reusing existing source files does not validate a fresh end-to-end reconstruction.

Generated outputs and caches are ignored by Git. Review the staged files before publishing; do not add identifiable demographics, raw recordings or restricted participant data.
