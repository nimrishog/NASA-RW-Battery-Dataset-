# Data availability and provenance

## Source data

The RW9, RW10, and RW11 MATLAB files used in this study originate from the NASA Randomized Battery Usage dataset distributed through NASA's Prognostics Center of Excellence data repository and NASA Open Data resources. Users of the data should cite the original NASA dataset and the associated manuscript.

## Repository contents

This repository provides:

- the RW9-RW11 source MATLAB files under `data/raw_mat/`;
- extraction quality-control summaries under `data/raw_extraction_summary/`;
- the event-level modeling table under `data/modeling/`;
- trained model states, predictions, metrics, and analysis tables under `results/`; and
- scripts needed to reproduce the extraction, modeling, sensitivity analyses, explainability analyses, and data-driven manuscript figures.

The raw source data remain subject to the terms and policies of their original provider. The MIT License in this repository applies to the analysis code and does not relicense the NASA source data or third-party content.

## Recommended archival release

Before journal publication, create a versioned archival release through Zenodo or an equivalent research repository and obtain a DOI. The manuscript data-availability statement should cite both the original NASA dataset and the archived release of this code and processed data. A tagged GitHub release alone is useful for versioning but does not provide the same preservation guarantee as a DOI-backed archive.

## Suggested manuscript statement

The source data are available from the NASA Randomized Battery Usage dataset. The code, processed RW9-RW11 event-feature table, trained-model outputs, evaluation tables, and figure-generation resources supporting this study are available in the project repository at https://github.com/nimrishog/NASA-RW-Battery-Dataset-. A versioned archival DOI will be added to the final article record upon repository release.
