# CrossRE
This repository contains the data and code for the paper:

Elisa Bassignana and Barbara Plank. 2022. CrossRE: A Cross-Domain Dataset for Relation Extraction. In Findings of the Association for Computational Linguistics: EMNLP 2022.

## The CrossRE Dataset
The data for each split (train, dev, test) of each domain (news, Artificial Intelligence, literature, music, politics, natural science) is in `crossre_data`.

The data is in the json format:
```
{
  "doc_key": "...",

  "sentence": [
    "token0", "token1", "token2", ...
    ]

  "ner": [
    [id-start, id-end, entity-type],
    [...], 
    ...
    ]

  "relations": [
    [id_1-start, id_1-end, id_2-start, id_2-end, relation-type, Exp, Un, SA],
    [...], 
    ...
  ]
}
```

### Annotation Guidelines
The annotation guidelines can be found in `crossre_annotation/CrossRE-annotation-guidelines.pdf`.

We release the annotations from the last annotation round (Round 5) in `crossre_annotation/last_annotation_round`.

## CrossRE Baselines
### Setup
Install all the dependency packages using the command:
```
pip install -r requirements.txt
```

### Run Experiments
Reproduce the baseline using the command:
```
./run.sh
```
Remember to set `EXP_PATH` and the `DOMAIN` of interest.

### Predictions
We release our predictions in the `predictions` folder.

## Cite
If you use the data, guidelines, code from CrossRE, please include the following reference:
```
@inproceedings{bassignana-plank-2022-crossre,
    title = "Cross{RE}: A {C}ross-{D}omain {D}ataset for {R}elation {E}xtraction",
    author = "Bassignana, Elisa and Plank, Barbara",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```
