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

## Archival release

The verified `v1.0.1` release is permanently archived on Zenodo with DOI [10.5281/zenodo.23179284](https://doi.org/10.5281/zenodo.23179284). The GitHub repository remains the development location, while the Zenodo record is the citable, versioned research archive.

## Suggested manuscript statement

The source data are available from the NASA Randomized Battery Usage dataset. The code, processed RW9-RW11 event-feature table, trained-model outputs, evaluation tables, and figure-generation resources supporting this study are available in the project repository at https://github.com/nimrishog/NASA-RW-Battery-Dataset- and are permanently archived on Zenodo at https://doi.org/10.5281/zenodo.23179284.
