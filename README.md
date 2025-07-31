# FineCite: A Novel Approach for Fine-Grained Citation Context Analysis

![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Authors:**  
Lasse M. Jantsch, Dong-Jae Koh, Seonghwan Yoon, Jisu Lee, Anne Lauscher, Young-Kyoon Suh

**Paper:** [FineCite: A Novel Approach for Fine-Grained Citation Context Analysis](https://aclanthology.org/2025.findings-acl.1259/)

---

## Overview

**FineCite** is a research codebase and dataset repository accompanying our paper on citation context analysis (CCA), a field concerned with understanding the role and intent of citations in scientific writing.

While prior CCA approaches often limit citation context to a single sentence, **FineCite** introduces a *fine-grained and semantically motivated* definition of citation context. We argue that overly simplistic context boundaries restrict the richness of scientific discourse and may hinder accurate classification of citation intent or function.

To address this, we:

- Propose a comprehensive citation context definition grounded in semantic properties of the citing text.
- Release **FineCite**, a novel dataset of **1,056 manually annotated** citation contexts with fine-grained labels.
- Provide evaluation on established CCA benchmarks, showing up to **25% performance gains** over state-of-the-art models.

This repository contains the source code and data necessary to reproduce the experiments presented in the paper.

---

## Getting Started

Follow these steps to set up the environment and run the extraction and classification models for citation context analysis.

---

### 1. Install Dependencies

We recommend using a virtual environment. Then install the required packages:

```bash
pip install torch argparse python-dotenv transformers numpy pandas torchmetrics peft bitsandbytes
```

### 2. Configure Environment Variables

Create a .env file in the root directory and add the following paths (update them as necessary for your system):

```
FINECITE_PATH=/path/to/FineCite
DATA_DIR=/path/to/FineCite/data
CACHE_DIR=/path/to/FineCite/.cache
OUT_DIR=/path/to/FineCite/output
```

### 3. Train Extraction Model
You can train the extraction model using a script. Run the script with your desired arguments.

```bash
python train_extraction.py --model_name scibert --ext_type bilstm_crf --save_model
```

### 4. Train Classification Model

You can train the extraction model using a script. Run the script with your desired configuration. 

```bash
python train_classifier.py --model_name scibert --dataset acl-arc --cls_type weighted --save_model
```
---

## License

This project is licensed under the MIT License.

---

## Citation

If you use FineCite in your research, please cite our paper:

```
@inproceedings{jantsch-etal-2025-finecite,
    title = "{F}ine{C}ite: A Novel Approach For Fine-Grained Citation Context Analysis",
    author = "Jantsch, Lasse M.  and
      Koh, Dong-Jae  and
      Yoon, Seonghwan  and
      Lee, Jisu  and
      Lauscher, Anne  and
      Suh, Young-Kyoon",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1259/",
    pages = "24525--24542",
    ISBN = "979-8-89176-256-5",
}
```
