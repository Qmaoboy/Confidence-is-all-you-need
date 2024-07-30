# CAPR: Confidence-Aware Prompt Refinement for Dehallucination

## Structure
```
./
├── api_key.yml
├── base_work
├── CAPR
├── PACE
├── pre_experiment.ipynb
└── README.md
```
## Preprocessing
Please Download the .json File throught google drive link below and unzip to folder respsectively:
- [Google Link](https://drive.google.com/file/d/1iZJbT_majQFzFjEUSw6ZfFNMqZOwuZLB/view?usp=sharing)


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
