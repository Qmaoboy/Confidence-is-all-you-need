# CAPR: Confidence-Aware Prompt Refinement for Dehallucination

## Structure
```
├── api_key.yml
├── base_work
├── CAPR
├── PACE
├── pre_experiment.ipynb
└── README.md
```

## How to Run
### PACE
- Enter Folder **PACE**
```
python pipline.py
```
### CAPR
- Enter Folder **CAPR**
    - For **Inference**
        ```
        python infernce.py
        ```
    - For **Traning**
        ```
        python TRL_training.py
        ```
### Baseline
- Contain **Vanilla** / **Self-Polish** / **Textgrad** / **Rephrease and Response(RaR)**
    - Modify base_line.py for differnet baseline work
        ```
        python base_line.py
        ```
