import argparse,torch
from LLM_API import *
from prompt_strategy import prompter,Prompt_refiner
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
import multiprocessing as mp
from util import *
from evaluate_confidence_score import evaluate_score
import json,os
from tqdm import tqdm
import os,json
from datetime import datetime
from infer_snli_cls import snli_similarity
import yaml
logger = setup_logger(f'log/response_{datetime.now().strftime("%Y%m%d")}.log')

class Prompt_refiner:
    def __init__(self):
        self.system_prompt="This is a prompt rewrite task,Please rewrite the prompt to New_Prompt in json"
        self.confidence_define_prompt="Note: The confidence indicates how likely you think your Answer is true and correct,from 0.00 (worst) to 1.00 (best)"

    def get_prompt(self,Old_prompt,confidence,accuracy):

        user_prompt=f"Rewrite the old_Prompt and old_Prompt Score, then provide a New_prompt in json."
        input_text=f"\nold_Prompt:{Old_prompt['user_prompt']+Old_prompt['input_text']},\n\n old_Prompt_Score : confidence:{confidence:.3f}, accuracy:{accuracy:.4f}\n\nNew_Prompt: [Your New_Prompt here]"

        return {"system_prompt":self.system_prompt,'user_prompt':user_prompt,'input_text':input_text,"assit_prompt":self.confidence_define_prompt}



if __name__=="__main__":
    if os.path.isfile("api_key.yml"):
        with open("api_key.yml","r") as f:
            key=yaml.safe_load(f)

    # train_dataloader=qadataset_dataloader(dataset_path="natural_questions",split='train',batch_size=1).loader
    refinement_dataloader=eval_dataloader(dataset_path="response_result/20240529/natural_questions_gpt-3.5-turbo-0125_vanilla_QA_Cos_sim_No_Shuffle.json",batch_size=1,purpose='refine').loader
    p_=Prompt_refiner()
    lambda_v=0.5
    eval_acc=acc_metric('bertscore')
    for idx,(batch,answer,ground_truth) in enumerate(refinement_dataloader):

        Prompt=batch[0][3]
        Doument=batch[0][0]
        Similarity_Score=max(map(float,batch[0][1])) ## Similarity
        Confident_score=batch[0][2] ## Confidence
        Final_score=lambda_v*Confident_score+(1-lambda_v)*Similarity_Score

        acc_batch= eval_acc.compute_acc(answer,ground_truth)[0]
        # print(document)
        prompt=p_.get_prompt(Prompt,Final_score,acc_batch)
        # print(acc_batch)
        # print(prompt)

        llm_api=GPT_API("gpt-3.5-turbo-0125",key,"refine",prompt)
        result,indi_complete_tokens,indi_Prompt_tokens=llm_api.generate()
        print(result)
        break


