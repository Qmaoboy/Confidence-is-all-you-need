from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from util import Show_mean_result
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

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def load_from_pretrained(pretrained_model_path):
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(base_model_name,token=key['hugginface']["token"],torch_dtype=torch.float16,use_cache=True, device_map = device)
    model = PeftModel.from_pretrained(model, pretrained_model_path)
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_path,token=key['hugginface']["token"],torch_dtype=torch.float16,use_cache=True,device_map={"": current_device})


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,token=key['hugginface']["token"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print("="*50+"Load From Pretrained !!!"+"="*50)
    return model,tokenizer



def inference(dataset_path,Save_result_path,Agent_addres):
    torch.cuda.empty_cache()
    trian_batch_size=100
    pretrained_path=Agent_addres
    model,tokenizer=load_from_pretrained(pretrained_path)
    generation_kwargs = {
    "min_length": -1,
    'temperature': 1,
    # "max_length": 100,
    "max_new_tokens": 64,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    'no_repeat_ngram_size':1
    }


    Dataloader=eval_dataloader(dataset_path=dataset_path, batch_size=trian_batch_size, purpose='refine',tokenizer=tokenizer,shuffle=True)
    result={
        'ece':[],
        'ece_capr':[],
        'acc':[],
        'acc_capr':[],
        'ece_pace':[],
        'ece_pace_capr':[]
    }

    for idx,(prompt,instruct,instruct_token,ans,ground_Truth,ground_Truth_token,Confidence,Document) in enumerate(bar:=tqdm(Dataloader.testloader)):
        instruct_token.input_ids=list(map(lambda x:torch.tensor(x),instruct_token.input_ids))
        input_qeury_token=tokenizer(instruct,padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                    **input_qeury_token,
                    **generation_kwargs
                )
            response = [tokenizer.decode(r[len(i):],skip_special_tokens=True) for r,i  in zip(outputs, input_qeury_token.input_ids)]


        # print("*"*50+"Before"+"*"*50)
        Before_result_batch=Parallel_Environment(prompt,'gpt-3.5-turbo-0125')
        _,before_ece,before_origin_ece,before_acc = reward_function(Before_result_batch,ground_Truth,Document)

        for p_instruc,p in zip(response, prompt):
            p['Instruction']=str(p_instruc)

        After_result_batch=Parallel_Environment(prompt,'gpt-3.5-turbo-0125')
        _,after_ece,after_origin_ece,after_acc = reward_function(After_result_batch,ground_Truth,Document)

        result['acc']+=[i.item() for i in before_acc]
        result['acc_capr']+=[i.item() for i in after_acc]

        result['ece']+=[i.item() for i in before_origin_ece]
        result['ece_capr']+=[i.item() for i in after_origin_ece]

        result['ece_pace']+=[i.item() for i in before_ece]
        result['ece_pace_capr']+=[i.item() for i in after_ece]

        bar.set_description_str(f"ece:{np.mean(np.array(result['ece']))} ece_capr:{np.mean(np.array(result['ece_capr']))},acc:{np.mean(np.array(result['acc']))}acc_capr:{np.mean(np.array(result['acc_capr']))}")
        break

    with open(Save_result_path,'w+') as f:
        json.dump(result,f,indent=4)


if __name__=="__main__":
    deliminator='Test'
    Agent_addres='Agent_weight/PPO_Agent_06112149_5_0.0463'
    inference(f'response_result/20240601/din0s_asqa_gpt-3.5-turbo-0125_vanilla_Long_QA.json',f"din0s_asqa_{deliminator}_Vanilla.json",Agent_addres)
    Show_mean_result(f"din0s_asqa_{deliminator}_Vanilla.json")
