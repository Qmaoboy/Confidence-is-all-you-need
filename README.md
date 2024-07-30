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
## Preprocessing- Data Download
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
        - Setup the different ==Agent_addres== in **inference.py** for different setting:
            - For ASQA Dataset:
                - CAPR w/ PACE: **PPO_Agent_06122032_vanilla_f1_r12_withPACE_7_0.0012**
                - CAPR w/o PACE: **PPO_Agent_06122032_vanilla_f1_r11_withoutPACE_9_0.0009**
                - CAPR w/ ACC: **PPO_Agent_06122032_vanilla_f1_r12_OnlyReward_7_0.0007**
            - For TriviaQA Dataset:
                - CAPR w/ PACE: **PPO_Agent_06122032_vanilla_f1_r1_trivia_withPACE_7_0.0030**
        - Then Run the following bash command:
            ```
            python infernce.py
            ```
    - For **Traning**
        - Setup ==Training_Config== for training detail:
            ```
            Training_Config={
                "dataset_path":f'response_result/20240601/triviaQA_gpt-3.5-turbo-0125_vanilla_QA.json', ## Training Data
                'deliminator':"06122032_vanilla_f1_r1_trivia_withPACE", ## Save_File deliminator
                'max_epoch': 8, ## Training Epoch
                'trian_batch_size':8, ## Training Batch Size
                'Batch_accumulate_size':32 ## Training Batch Accumulate Size min : 128, Max: 64
                            }
            ```
        - Then Run the following bash command:
            ```
            python TRL_training.py
            ```
### Baseline
- Contain **Vanilla** / **Self-Polish** / **Textgrad** / **Rephrease and Response(RaR)**
    - Modify base_line.py for differnet baseline work
        ```
        python base_line.py
        ```
