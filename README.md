# Melanoma Pathology NLP

This is the code for melanoma diagnosis and feature extraction from free-text pathology reports, using the MedspaCy NLP system. A variant of this pipeline intended for provider notes is also included, as well as the guideline and schema used for manual annotation.

# Project description:

The goal of the current application is to:

- Identify melanoma diagnosis and information related to the diagnosis, especially those related to TNM staging. Extracted concepts include: melanoma histology, sample site, metastasis, mitotic index, clark level, breslow depth, ulceration, and deep margin involvement.
- Assess context for currency and
- Relates concepts together and aggregate information to form distinct, biopsy-level, document-level, and patient-level analytic tables.

## MedspaCy pipeline

Defined using the build_nlp() method. This system employs a series of sequential pipeline components for matching and relating relevant concepts. More specifically, the melanoma pipeline:

1) chunks text into smaller units (i.e. tokenization),
2) tags relevant terms, such as measurement values, histology terms, and anatomic sites,
3) uses combinations of tagged terms to mark relevant histologies (i.e. Named Entity Recognition), and
4) uses combinations of tagged terms to mark related concepts within a dynamic scope and direction of the histologic entities, such as topography, depth, temporal, negation, and uncertainty information.

The included postprocessing package will further transform the MedspaCy output. Given the returned list of spaCy documents, the transform method will:
5) extracts entities and related features and parse output for measurement values
6) aggregate all histologies and associated features at a biopsy sample  level, and incorporate sample level logic for combining and interpreting results, and
7) aggregate information for document-level analysis.
*Note that the postprocessing may be merged into the medspacy pipeline on a future iteration

Rules for all taggers, named entity matchers, and context matchers are included in the 'resources' folder and are organized by concepts.

## Output Types

Data goes through several transformations. The basic, raw output of the system and initial mappings are within the transform_dict[] dataframe. Data is aggregated based on feature-histology relationships, and histology-topography relationships. Some logic is also applied on a sample level. The below output table shows the table for specimen level.

| Output Type             | Description                                                                                                                                                                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DocID                   | Document identifier                                                                                                                                                                                                                                                                                             |
| Topography              | NLP-extracted specimen location. Note that NLP only matches topography when there is an associated tumor, diagnosis, or 'pending' term. If there is a tumor/diagnosis/pending term found and no topography found, this will be 'None'                                                                           |
| SkinTopography          | 1 if the Topography was identified as a skin sample. If topography uses a non-skin anatomy or is 'None', this is 0                                                                                                                                                                                              |
| Breslow_depth_mm        | Extracted breslow depth (normalized to millimeter measurement). 'None' if no breslow depth found.                                                                                                                                                                                                               |
| clark_annot             | Depth defined by clark level                                                                                                                                                                                                                                                                                    |
| metastasis_             | 1 if explicit metastasis term was found and is associated with a melanoma dx, OR there is a melanoma with a non-skin topography. 0 if no explicit metastasis term found or metastasis term is associated with historical or negated melanoma diagnosis.                                                         |
| ulceration_status       | Ulceration status. Possible values: 'absent','present','impending','indeterminate'. 'None' if no ulceration concepts matched.                                                                                                                                                                                   |
| cutaneous               | 1 if any of the following melanoma diagnoses are found: 'desmoplastic','epithelioid','lentigo_maligna','melanoma_unspecified','nevoid','nodular','not_otherwise_specified', 'superficial_spreading','spindle_cell','spitzoid', 'letiginous','acral','amelanotic','hutchinsons_melanotic_freckle','balloon_cell' |
| mucosal                 | 1 if there is an explicit 'mucosal' melanoma diagnosis                                                                                                                                                                                                                                                          |
| in_situ                 | 1 if there is in-situ melanoma                                                                                                                                                                                                                                                                                  |
| ocular                  | 1 if there is an ocular melanoma diagnosis. This includes cases where explicit 'ocular' term is found, 'choroidal' diagnosis is found, or an unspecified melanoma is associated with topography from the eye                                                                                                    |
| Mitoses                 | Mitotic index value. Otherwise, descriptive result if found. Otherwise, 'None'.                                                                                                                                                                                                                                 |
| deep_margin_involvement | 1 if a deep transection or deep margin involvement is extracted, OR if there is evidence that breslow depth is greater than the excision.                                                                                                                                                                       |
| non_melanoma_dx         | 1 if there was a non-melanoma diagnosis matched. Note that the NLP only handles a subset of other possible skin diagnoses, and does not focus on feature extraction for these other diagnoses.                                                                                                                  |
| cancer_negated          | 1 if a cancer (including melonoma) term is negated. Note that this is NOT mutually exclusive to other positive melanoma dx's.                                                                                                                                                                                   |
| pending                 | 1 if 'pending' concept matched. This may indicate data is lacking for a specimen in the document                                                                                                                                                                                                                |
| Topography_start_span   | Start offset for Topography                                                                                                                                                                                                                                                                                     |
| Topography_end_span     | End offset for Topography                                                                                                                                                                                                                                                                                       |

# Usage

This system is based on MedspaCy version 1.3.1, which is an extension of spaCy version 3.7.5. For more information on MedspaCy, see https://github.com/medspacy/medspacy.

To use this pipeline: Follow instructions to install MedspaCy, run the build_nlp() method found in notebooks/melanoma_nlp.py to assemble the components and rules, use the run_nlp() method to run the nlp system on an list-like object. See the tutorial for sample implementation.
