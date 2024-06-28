from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from trl import PPOConfig,PPOTrainer
import torch.multiprocessing as t_mp
import torch.distributed as dist
from torch.optim.lr_scheduler import ConstantLR,ExponentialLR,SequentialLR,StepLR
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from RL_env import Environment,reward_function,rl_writer,Parallel_Environment
import glob,os,torch,yaml
from huggingface_hub import login
import json
if os.path.isfile("api_key.yml"):
    with open("api_key.yml","r") as f:
        key=yaml.safe_load(f)

if os.path.isfile("default_config.yaml"):
    with open("default_config.yaml","r") as f:
        ac_config=yaml.safe_load(f)

login(token=key['hugginface']["token"])

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def Get_auroc(accuracy,confidence_scores):
    y_true=np.where(np.array(accuracy) < 0.3,0,1)
    return roc_auc_score(np.array(y_true), np.array(confidence_scores))


class inference:
    def __init__(self,pretrained_path,dataset_path,Save_result_path) -> None:
        torch.cuda.empty_cache()
        self.result={}
        self.model,self.tokenizer=self.load_from_pretrained(pretrained_path)
        self.dataset_path=dataset_path
        self.Save_result_path=Save_result_path
        self.generation_kwargs = {
        "min_length": -1,
        'temperature': 1,
        "max_length": 256,
        # "max_new_tokens": 96, # before : 128
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": self.tokenizer.eos_token_id,
        'no_repeat_ngram_size':4
        }

    def load_from_pretrained(self,pretrained_model_path):
        base_model_name = "meta-llama/Llama-2-7b-chat-hf"
        model = AutoModelForCausalLM.from_pretrained(base_model_name,token=key['hugginface']["token"],torch_dtype=torch.bfloat16,use_cache=True, device_map = device)
        model = PeftModel.from_pretrained(model, pretrained_model_path)
        # model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_path,token=key['hugginface']["token"],torch_dtype=torch.float16,use_cache=True,device_map={"": current_device})

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,token=key['hugginface']["token"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        print("="*50+"Load From Pretrained !!!"+"="*50)
        return model,tokenizer

    def get_result(self,key,prompt,instruction,ground_Truth,Document):
        '''
        Prompt Contain:
            system_prompt
            Instruction
            question
            input_text
            assit_prompt
        '''
        for idx,p_instruc in enumerate(instruction):
            prompt[idx]['Instruction']=p_instruc
            prompt[idx]['system_prompt']="This is a Long form [generation QA task, please answer the Question base on the Instruction to the question and confidence to the Answer in json."

        result_batch=Parallel_Environment(prompt,'gpt-3.5-turbo-0125')
        _,pace_ece,Verbalized_ece,Accuracy,Pace_Conf,Verbalized_conf = reward_function(result_batch,ground_Truth,Document)

        self.result[key]={
            'pace_ece':[i.item() for i in pace_ece],
            'Verbalized_ece':[i.item() for i in Verbalized_ece],
            'Accuracy':[i.item() for i in Accuracy],
            'Pace_Conf':[i.item() for i in Pace_Conf],
            "Verbalized_conf":[i.item() for i in Verbalized_conf],
            'Auroc':[Get_auroc([i.item() for i in Accuracy],[i.item() for i in Verbalized_conf])],
            'PACE_Auroc':[Get_auroc([i.item() for i in Accuracy],[i.item() for i in Pace_Conf])]
        }

    def generate_result(self,instruct):
        input_qeury_token=self.tokenizer(instruct,padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                    **input_qeury_token,
                    **self.generation_kwargs
                )
            response = [self.tokenizer.decode(r[len(i):],skip_special_tokens=True) for r,i  in zip(outputs, input_qeury_token.input_ids)]

        return response

    def get_inference(self,):

        trian_batch_size=50
        Dataloader=eval_dataloader(dataset_path=self.dataset_path, batch_size=trian_batch_size, purpose='refine',tokenizer=self.tokenizer,shuffle=True)

        for idx,(prompt,instruct,instruct_token,ans,ground_Truth,ground_Truth_token,Confidence,Document) in enumerate(bar:=tqdm(Dataloader.testloader)):
            # instruct_token.input_ids=list(map(lambda x:torch.tensor(x),instruct_token.input_ids))
            response=self.generate_result(instruct)
            # bar.set_description_str(f"origin Prompt")
            # self.get_result('origin',prompt,[""]*len(prompt),ground_Truth,Document)
            bar.set_description_str(f"refine Prompt")
            self.get_result('refine',prompt,response,ground_Truth,Document)
            break
        self.Save_File()

    def Save_File(self):
        with open(self.Save_result_path,'w+') as f:
            json.dump(self.result,f,indent=4)

def Show_mean_result(key,Save_result_path):
    if os.path.isfile(Save_result_path):
        with open(Save_result_path,'r') as f:
            result=json.load(f)
        for k,v in result.items():
            if k==key:
                print(k)
                for k1,v1 in v.items():
                    print(f"\t{k1} : {np.mean(np.array(v1)):.6f}")


if __name__=="__main__":
    ## Setting
    deliminator='r11_with_vanilla'
    Agent_addres='Agent_weight/PPO_Agent_06122032_vanilla_f1_r11_9_0.0009'
    dataset_path=f'response_result/20240601/din0s_asqa_gpt-3.5-turbo-0125_vanilla_Long_QA.json'
    Save_result_path=f"din0s_asqa_{deliminator}.json"
    ##
    inf=inference(Agent_addres,dataset_path,Save_result_path)
    inf.get_inference()
    # Show_mean_result("origin",Save_result_path)
    Show_mean_result("refine",Save_result_path)
