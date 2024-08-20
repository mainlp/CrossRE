# CrossRE
This repository contains the data and code for the papers:

`crossre_data` Elisa Bassignana and Barbara Plank. 2022. [CrossRE: A Cross-Domain Dataset for Relation Extraction.](https://aclanthology.org/2022.findings-emnlp.263.pdf) In Findings of the Association for Computational Linguistics: EMNLP 2022.

`multi-crossre_data` Elisa Bassignana, Filip Ginter, Sampo Pyysalo, Rob van der Goot, and Barbara Plank. 2023. [Multi-CrossRE: A Multi-Lingual Multi-Domain Dataset for Relation Extraction.](https://openreview.net/pdf?id=G8pAo0rvbh) In Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa 2023).

`crossre_extension` Elisa Bassignana, Viggo Unmack Gascou, Frida Nøhr Laustsen, Gustav Kristensen, Marie Haahr Petersen, Rob van der Goot and Barbara Plank. 2024. [How to Encode Domain Information in Relation Classification.](https://aclanthology.org/2024.lrec-main.728.pdf) In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024).

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
If you use the data, guidelines, code from CrossRE, Multi-CrossRE, CrossRE 2.0, please include the following references:
```
@inproceedings{bassignana-plank-2022-crossre,
    title = "{C}ross{RE}: A Cross-Domain Dataset for Relation Extraction",
    author = "Bassignana, Elisa  and
      Plank, Barbara",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.263",
    pages = "3592--3604",
    abstract = "Relation Extraction (RE) has attracted increasing attention, but current RE evaluation is limited to in-domain evaluation setups. Little is known on how well a RE system fares in challenging, but realistic out-of-distribution evaluation setups. To address this gap, we propose CrossRE, a new, freely-available cross-domain benchmark for RE, which comprises six distinct text domains and includes multi-label annotations. An additional innovation is that we release meta-data collected during annotation, to include explanations and flags of difficult instances. We provide an empirical evaluation with a state-of-the-art model for relation classification. As the meta-data enables us to shed new light on the state-of-the-art model, we provide a comprehensive analysis on the impact of difficult cases and find correlations between model and human annotations. Overall, our empirical investigation highlights the difficulty of cross-domain RE. We release our dataset, to spur more research in this direction.",
}
```
```
@inproceedings{bassignana-etal-2023-multi,
    title = "Multi-{C}ross{RE} A Multi-Lingual Multi-Domain Dataset for Relation Extraction",
    author = "Bassignana, Elisa  and
      Ginter, Filip  and
      Pyysalo, Sampo  and
      Goot, Rob  and
      Plank, Barbara",
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.9",
    pages = "80--85",
    abstract = "Most research in Relation Extraction (RE) involves the English language, mainly due to the lack of multi-lingual resources. We propose Multi-CrossRE, the broadest multi-lingual dataset for RE, including 26 languages in addition to English, and covering six text domains. Multi-CrossRE is a machine translated version of CrossRE (Bassignana and Plank, 2022), with a sub-portion including more than 200 sentences in seven diverse languages checked by native speakers. We run a baseline model over the 26 new datasets and{--}as sanity check{--}over the 26 back-translations to English. Results on the back-translated data are consistent with the ones on the original English CrossRE, indicating high quality of the translation and the resulting dataset.",
}
```
```
@inproceedings{bassignana-etal-2024-encode,
    title = "How to Encode Domain Information in Relation Classification",
    author = "Bassignana, Elisa  and
      Gascou, Viggo Unmack  and
      Laustsen, Frida N{\o}hr  and
      Kristensen, Gustav  and
      Petersen, Marie Haahr  and
      van der Goot, Rob  and
      Plank, Barbara",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.728",
    pages = "8301--8306",
    abstract = "Current language models require a lot of training data to obtain high performance. For Relation Classification (RC), many datasets are domain-specific, so combining datasets to obtain better performance is non-trivial. We explore a multi-domain training setup for RC, and attempt to improve performance by encoding domain information. Our proposed models improve {\textgreater} 2 Macro-F1 against the baseline setup, and our analysis reveals that not all the labels benefit the same: The classes which occupy a similar space across domains (i.e., their interpretation is close across them, for example {``}physical{''}) benefit the least, while domain-dependent relations (e.g., {``}part-of{''}) improve the most when encoding domain information.",
}
```
