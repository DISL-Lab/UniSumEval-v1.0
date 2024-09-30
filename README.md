<h1 align="center">
UniSumEval
</h1>


<h3 align="center">
Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs
</h3>



<div align="center">
  <p>Authors: Yuho Lee, Taewon Yun, Jason Cai, Hang Su, Hwanjun Song</p>
  <a href="https://arxiv.org/abs/1234.56789">
  <img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat-square" alt="arXiv">
  </a>
  <a href="https://disl-lab.github.io/">
  <img src = "https://img.shields.io/badge/Data%20Intelligence%20System%20Lab-B552CB.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iNTQ0LjAwMDAwMHB0IiBoZWlnaHQ9IjU0MS4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDU0NC4wMDAwMDAgNTQxLjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsNTQxLjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTExNDggNDQzOSBsLTMgLTk1MSAtNzMgLTE5IGMtNTQwIC0xMzcgLTk1NSAtNTk0IC0xMDM2IC0xMTQxIC0xMTEKLTc0NyAzODUgLTE0NDQgMTEyOSAtMTU4NCA2NyAtMTIgMTY4IC0xOCAzOTMgLTIxIGwzMDIgLTUgMCAtMzU1IDAgLTM1NSA1NjgKNSBjNDQ1IDMgNTkwIDggNjcyIDIwIDQ4NyA3MiA5MjYgMjYxIDEzMDAgNTU3IDEyNCA5OCAzMzggMzEwIDQzNCA0MzAgMjc3CjM0NyA0NzggNzk2IDU1MCAxMjMwIDQxIDI0NSA0NiA1OTUgMTEgODI4IC02OSA0NzQgLTI2OCA5MzUgLTU2MSAxMzAyIC04NwoxMDkgLTMxNSAzMzcgLTQyNCA0MjQgLTMxNSAyNTEgLTY5MSA0MzEgLTEwOTEgNTIyIC0yNTUgNTcgLTI0OSA1NyAtMTI0NiA2MQpsLTkyMyA0IC0yIC05NTJ6IG0xOTQyIDU5NyBjNDEzIC02OSA3OTYgLTIzNyAxMTA1IC00ODQgODIzIC02NjAgMTExNCAtMTc1Nwo3MjYgLTI3MzUgLTE3MCAtNDI3IC00OTcgLTgyNSAtODk2IC0xMDkxIC0zMDUgLTIwNCAtNjg4IC0zMzkgLTEwNTkgLTM3NSAtNjAKLTYgLTI1OSAtMTEgLTQ0MyAtMTEgbC0zMzMgMCAwIDE5MCAwIDE5MCAyOTggMCBjNDQwIDAgNjE4IDIyIDg3MCAxMDYgMjg0IDk1CjUyMyAyNDAgNzQ1IDQ1MyAyNjAgMjUwIDQzNSA1NDYgNTM3IDkwOSA4NiAzMDQgODcgNjg4IDQgMTAwOCAtMjIzIDg1OSAtMTAwMgoxNDcxIC0xODc5IDE0NzcgLTk4IDAgLTE2MiAtNSAtMjE0IC0xNiAtMzE5IC03MyAtNTYzIC0zMDAgLTY1NyAtNjE0IC0yNSAtODEKLTI3IC0xMDYgLTMyIC0zMDUgbC00IC0yMTggLTEzNyAwIGMtNzUgMCAtMTYwIC0zIC0xODggLTYgbC01MyAtNyAwIDc3NyAwCjc3NyA3NTMgLTQgYzY2MCAtMyA3NjUgLTUgODU3IC0yMXogbS0xNzAgLTcwMSBjOTQ1IC0xMTAgMTU5NiAtOTg0IDE0MzQKLTE5MjUgLTE5IC0xMTQgLTc5IC0yOTkgLTEzMiAtNDEyIC0yMTAgLTQ0OCAtNjE2IC03NzkgLTExMDcgLTkwMiAtMTM3IC0zNQotMjkwIC00NiAtNjExIC00NiBsLTMxNCAwIDAgMTk1IDAgMTk1IDMyOSAwIGMzOTkgLTEgNDc2IDkgNjc4IDg2IDIyNSA4NiA0NDcKMjY2IDU4OCA0NzggMjE1IDMyNCAyNjcgNzQwIDEzOCAxMTEwIC00MiAxMjEgLTY5IDE3NSAtMTQxIDI4NiAtNjggMTA0IC0yNDAKMjc4IC0zMzcgMzQyIC0yMTIgMTM5IC00NTkgMjE4IC02ODAgMjE4IC02OSAwIC04NyAtNCAtMTE1IC0yMiAtNzggLTUzIC0xMDAKLTEzOCAtNTggLTIxNyAzNCAtNjMgNzUgLTgxIDIwMyAtOTEgMzY0IC0zMCA2NTYgLTI0MCA4MDAgLTU3NyA0NyAtMTEwIDY4Ci0yMTcgNjkgLTM1MyAxIC0zNjIgLTIwMCAtNjgyIC01MjggLTg0MCAtMTc2IC04NCAtMjY4IC05OCAtNjczIC05OSBsLTI3MyAtMgowIDEwNTkgYzAgOTA1IDIgMTA2NiAxNSAxMTE1IDUxIDE5OCAyMTggMzU5IDQxNSA0MDEgODYgMTkgMTQ5IDE5IDMwMCAxegptLTEwNjIgLTE4NTcgbC0zIC03MTMgLTU3IDMgYy0xMzMgNiAtMjUzIDg5IC0yOTYgMjA1IC0yMiA1NyAtMjIgNjggLTIyIDYzNwpsMCA1ODAgMTkwIDAgMTkwIDAgLTIgLTcxMnogbS03MDggMTA1IGMwIC02MzggMCAtNjM2IDcyIC03ODMgMzMgLTY3IDYwIC0xMDIKMTIzIC0xNjUgMTIzIC0xMjQgMjQ5IC0xODEgNDIwIC0xOTQgbDkwIC02IDMgLTE5MyAyIC0xOTQgLTMwMiA1IGMtMjI0IDQKLTMyMCA5IC0zNjggMjAgLTQyOSAxMDIgLTc1MiA0MzYgLTgyNSA4NTQgLTE5IDEwOCAtMTkgMjc0IC0xIDM3OCA2NyAzNzggMzQzCjY5OCA3MTEgODI0IDMzIDExIDYzIDIwIDY4IDIxIDQgMCA3IC0yNTUgNyAtNTY3eiIvPgo8L2c+Cjwvc3ZnPgo=&style=flat-square">
  </a>

  <img src = "https://img.shields.io/badge/KAIST%20GSDS-087CFA.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iMzk3LjAwMDAwMHB0IiBoZWlnaHQ9IjEyNC4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDM5Ny4wMDAwMDAgMTI0LjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsMTI0LjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTM3NyAxMTA1IGMtMTE2IC0zMiAtMjAyIC0xMzEgLTIyNiAtMjYwIC0xNCAtNzQgLTE0IC0yNjAgLTEgLTM0NAoyMCAtMTE3IDg2IC0yMDggMTg0IC0yNTIgMTIwIC01MiA0NjcgLTUwIDU2MCA1IDkwIDUzIDExNiAxMjEgMTE2IDMwOCBsMCAxMjgKLTE4NSAwIC0xODUgMCAwIC05MCAwIC05MCA4NiAwIDg3IDAgLTYgLTQyIGMtOSAtNjQgLTE5IC02OCAtMTk3IC02OCAtMTQ2IDAKLTE1OSAyIC0xOTAgMjMgLTY2IDQ0IC03NSA3MyAtNzUgMjQ3IDAgMTcxIDggMTk5IDY4IDI0NCAyNyAyMCA0MiAyMSAyNzggMjQKbDI0OSAzIDAgODkgMCA5MCAtMjU3IC0xIGMtMTgxIC0xIC0yNzIgLTUgLTMwNiAtMTR6Ii8+CjxwYXRoIGQ9Ik0yMDYwIDY3MCBsMCAtNDUwIDI2OCAwIGMzNDEgMCAzOTEgMTIgNDc3IDExMiA2NyA3OSA3OSAxMjkgNzkgMzM4CjAgMjA5IC0xMiAyNTkgLTc5IDMzOCAtODYgMTAwIC0xMzYgMTEyIC00NzcgMTEyIGwtMjY4IDAgMCAtNDUweiBtNTYxIDI0NwpjNjAgLTQwIDY5IC03MiA2OSAtMjQ3IDAgLTE3NSAtOSAtMjA3IC02OSAtMjQ3IC0zMiAtMjIgLTQ0IC0yMyAtMjAyIC0yMwpsLTE2OSAwIDAgMjcwIDAgMjcwIDE2OSAwIGMxNTggMCAxNzAgLTEgMjAyIC0yM3oiLz4KPHBhdGggZD0iTTEyNzcgMTA5MyBjLTY2IC0yMiAtMTQxIC05MSAtMTY0IC0xNTEgLTQyIC0xMTIgLTMgLTI0NyA5MCAtMzA5IDcwCi00NyAxMDQgLTU0IDI4NSAtNjAgMTQ5IC01IDE2NiAtOCAxODkgLTI3IDM2IC0zMSAzNSAtOTIgLTIgLTEyMyAtMjYgLTIzIC0zMAotMjMgLTI4NiAtMjMgbC0yNTkgMCAwIC05MSAwIC05MCAzMDggMyAzMDcgMyA1NCAzMCBjODggNDkgMTMzIDEyNiAxMzQgMjMxIDEKMTE3IC03NyAyMjEgLTE5MSAyNTQgLTIwIDYgLTExMSAxNCAtMjAyIDE3IC0xNTQgNiAtMTY1IDggLTE4NyAyOSAtMjUgMjYgLTMxCjc4IC0xMCAxMDUgMjcgMzYgNTMgMzkgMjg2IDM5IGwyMzEgMCAwIDkwIDAgOTAgLTI2NyAtMSBjLTIxOSAwIC0yNzcgLTMgLTMxNgotMTZ6Ii8+CjxwYXRoIGQ9Ik0zMTczIDEwOTYgYy0xMzcgLTQzIC0yMTIgLTE3MCAtMTgzIC0zMDkgMTIgLTYwIDUxIC0xMjIgOTcgLTE1NSA1OQotNDIgMTIxIC01NCAyOTEgLTYwIDE5MyAtNiAyMTIgLTEzIDIxMiAtODcgMCAtMzUgLTUgLTQ4IC0yNiAtNjQgLTI1IC0yMCAtMzkKLTIxIC0yOTAgLTIxIGwtMjY0IDAgMCAtOTEgMCAtOTAgMzA4IDMgYzI4OCAzIDMxMCA0IDM1MyAyNCAyNSAxMiA2MCAzNyA3OAo1NSAxMTAgMTE0IDk0IDMxMSAtMzIgMzk2IC02OSA0NiAtMTAyIDUzIC0yODcgNjAgLTE1NCA2IC0xNjUgOCAtMTg3IDI5IC0xNAoxNCAtMjMgMzUgLTIzIDUzIDAgODUgMTkgOTEgMzAwIDkxIGwyMzAgMCAwIDkwIDAgOTAgLTI2NyAtMSBjLTE5MSAwIC0yODAgLTQKLTMxMCAtMTN6Ii8+CjwvZz4KPC9zdmc+Cg==&style=flat-square">
  
<p align="center">
  <img src="https://github.com/DISL-Lab/UniSumEval/assets/123562613/cf6c27a7-2de7-4e60-833c-69555e75e51c" alt="" width="500">
</p>

</div>

---

This is the official github repository for "UniSumEval: Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs - to be changed (hyperlink)"

---

# 📖 Data Description

We provide samples in Unisumeval, which is available in [`data/unisumeval_faithfulness.jsonl`](data/unisumeval_faithfulness.jsonl) and [`data/unisumeval_keyfact.jsonl`](data/unisumeval_keyfact.jsonl). 

If you have any questions about our dataset, please contact us at ytaewon0415@kaist.ac.kr or kaist.disl.lab@gmail.com.

<br/> 

## Metadata
The metadata provides basic information about UniSumEval. All attributes are included in every file. Below is a description of each attribute. 

| Field       | Description                                |
|-------------|--------------------------------------------|
| uid         | Dataset unique ID (DISL_XXX)               |
| doc_id      | ID in source dataset                       |
| source      | Source dataset name                        |
| domain      | Domain                                     |
| length      | Short / Long                               |
| input_type  | Dialogue / Non-Dialogue                    |
| redact_type | unredact / redact                          |
| model       | Summarizer name                            |
| reference   | Reference summary in source dataset        |
| summary     | Model summary (after JSON parsing)         |
| sentences   | Summary splited by sentence                |

<br/> 

## Fact verification annotations
[`data/unisumeval_faithfulness.jsonl`](data/unisumeval_faithfulness.jsonl) contain faithfulness evaluations by human annotators in mturk. The followings are descriptions about the attributes in JSON

| Field                                    | Description                                                               |
|------------------------------------------|---------------------------------------------------------------------------|
| fact_verification_label                  | Fact verification annotations about binary label (1 = hallu) |
| fact_verification_error_type             | Fact verification annotations about error type      |
| faithfulness_score     | Faithfulness score based on human annotation                               |

<br/> 

## Key-fact validation annotations
[`data/unisumeval_keyfact.jsonl`](data/unisumeval_keyfact.jsonl) contain key-facts validation results by human annotators in mturk. The followings are descriptions about the attributes in JSON

| Field                           | Description                                                      |
|---------------------------------|------------------------------------------------------------------|
| pred_keyfact                    | Potential keyfact after machine cross-validation                |
| keyfact_validation_label        | Keyfact validation annotation             |
| keyfact                         | Keyfact after keyfact validation                                 |

<br/> 

## Key-fact alignment annotations
[`data/unisumeval_keyfact.jsonl`](data/unisumeval_keyfact.jsonl) contains completeness and conciseness evaluations by aligning summary sentences with key facts. The followings are descriptions about the attributes in JSON

| Field                                         | Description                                                      |
|-----------------------------------------------|------------------------------------------------------------------|
| keyfact_label                                 | The majority class of keyfact alignment annotations for each keyfact |
| completeness                                  | Completeness score based on human annotation                     |
| sentence_label                                | The majority class of keyfact alignment annotations for each sentence |
| conciseness                                   | Conciseness score based on human annotation                      |

<br/> 

# 📝 Annotation Templates
Our annotation interfaces can be found in [`annotation/layout`](annotation/layout). we also include html files, which are available in the [`annotation/templete`](annotation/templete) 

<br/> 

# 📋 Reproduce the Main Table of the Paper (Evaluators Results)
```bash
cd ../UniSumEval
python reproduce/evaluators_benchmark.py --input_path data/unisumeval_faithfulness.json --output_path results/evaluators_benchmark_faithfulness.csv --score_dict machine_evaluation_results_faithfulness --gt_score faithfulness_score
```

<br/> 

# 📄 Main Results

<p align="center">
  <img src="https://github.com/DISL-Lab/UniSumEval/assets/123562613/0e420107-e549-41c7-83bd-c2dc3677798f" alt="">
</p>

- This figure shows the overall performance rankings aggregated across the all text domains, types, and lengths in UniSumEval. The proprietary LLMs notably outperform the non-LLMs and open-source LLMs across the all aspects. **Proprietary LLMs also exhibit no statistical relationship between faithfulness and abstractiveness - contradicting recent findings that summaries with higher abstractiveness are more prone to trigger hallucination.**

- **Additionally, the input contexts—such as input domain, type, and length—can impact the performance of each summarizer. These observations are detailed in our paper.**


<br/> 

# 🖇️ Citation

Please consider citation if our paper is useful in your research.
```BibTeX
@inproceedings{lee2024unisumeval,
  title={UniSumEval: Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs},
  author={Lee, Yuho and Yun, Taewon and Cai, Jason and Su, Hang and Song, Hwanjun},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  year={2023}
}
```

# 🤗 Acknowledgement
This work was supported by Korea Advanced Institute of Science & Technology Graduate School of Data Science (KAIST GSDS). 

###### *This work was done @ KAIST Data Intelligence System Lab*
