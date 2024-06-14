from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
# from peft import PeftModel, PeftConfig, get_peft_model
from accelerate import Accelerator
import peft
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
# from torchtune.utils import set_seed
from tqdm import tqdm
from util import Show_mean_result
from trl import PPOConfig,PPOTrainer
import torch.multiprocessing as t_mp
import torch.distributed as dist
from random import randint
from inference import inference
from torch.optim.lr_scheduler import ConstantLR,PolynomialLR,SequentialLR,StepLR
import json
from RL_env import Environment,reward_function,rl_writer,Parallel_Environment
import glob,os,torch,yaml
from huggingface_hub import login

if os.path.isfile("api_key.yml"):
    with open("api_key.yml","r") as f:
        key=yaml.safe_load(f)

if os.path.isfile("default_config.yaml"):
    with open("default_config.yaml","r") as f:
        ac_config=yaml.safe_load(f)

login(token=key['hugginface']["token"])

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '49527'

    # initialize the process group
    dist.init_process_group("qmao_gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def trainer(world_size, max_epoch, model, tokenizer,Dataloader,generation_kwargs,writer):

    # rank=os.environ['LOCAL_RANK']
    # print(f"Running DDP on rank {rank}.")


    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # old_reward=0
    ## Start To Train With Loaded Dataset
    ppo_trainer=model
    update_count=0
    update_freq=15
    # stream = torch.cuda.current_stream(rank)
    for epoch in (t:=tqdm(range(max_epoch), "epoch: ")):
        for prompt,instruct,instruct_token,ans,ground_Truth,ground_Truth_token,Confidence,Document in (bar:=tqdm(Dataloader.trainloader)):
            instruct_token.input_ids=list(map(lambda x:torch.tensor(x),instruct_token.input_ids))
            query_tensors=instruct_token.input_ids
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensor=query_tensors, return_prompt=True, **generation_kwargs)
            instruct_token['response'] = tokenizer.batch_decode(response_tensors)
            instruct_token['query'] = instruct

            response = [tokenizer.decode(r[len(i):],skip_special_tokens=True) for r,i  in zip(response_tensors, instruct_token.input_ids)]
            assert len(prompt)==len(response)
            print("\n"+"*"*25+"Start Generate Instruction"+"*"*25+"\n")
            ## replace generated Instruction
            show_index=randint(0,len(response)-1)
            print(f"Sample id {show_index}:")
            print(f"Instruct:\n{prompt[show_index]['Instruction']}")
            print(f"Question:\n{prompt[show_index]['Question']}\n")
            print(f"After:\n{response[show_index]}")

            for idx,p_instruc in enumerate(response):
                prompt[idx]['Instruction']=str(p_instruc)
                prompt[idx]['system_prompt']="This is a Long form generation QA task, please answer the Question base on the Instruction to the question and confidence to the Answer in json."

            ## Environment Get Answer and Confidence
            print("\n"+"*"*25+"Start Running Environment"+"*"*25+"\n")

            result_batch=Parallel_Environment(prompt,'gpt-3.5-turbo-0125')

            print("\n"+"*"*25+"Start Get Reward"+"*"*25+"\n")
            # print(result_batch)
            #### Compute reward score
            Reward,ece,origin_ece,acc,pace_conf,conf = reward_function(result_batch,ground_Truth,Document)
            # old_reward=torch.mean(torch.stack(Reward)).item()
            #####################

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, Reward)
            ppo_trainer.log_stats(stats, instruct_token, Reward)
            #####################

            ### Record the result
            writer.get([stats['ppo/val/error']],measure='mean',key='Val_error')
            writer.get([stats['ppo/learning_rate']],measure='mean',key='learning_rate')
            writer.get([stats['ppo/loss/value']],measure='mean',key='loss_value')
            writer.get([stats['ppo/loss/total']],measure='mean',key='loss_total')
            writer.get([stats['ppo/loss/policy']],measure='mean',key='loss_policy')
            writer.get([stats['ppo/policy/approxkl']],measure='mean',key='ApproxKL')
            writer.get([stats['ppo/policy/policykl']],measure='mean',key='policykl')
            writer.get(Reward,measure='mean',key='reward')
            writer.get(acc,measure='mean',key='Accuracy')
            writer.get(ece,measure='mean',key='ECE')
            writer.get(pace_conf,measure='mean',key='Pace_Verbalized_Confidence')
            writer.get(conf,measure='mean',key='Verbalized_Confidence')
            writer.write()
            #####################
            print(f"\nTraining:{epoch}/{max_epoch}\nloss_value {writer.data_value_writer['loss_value'][-1]:.4f},\nloss_total {writer.data_value_writer['loss_total'][-1]:.4f},\nAccuracy {writer.data_value_writer['Accuracy'][-1]:.4f}\nECE {writer.data_value_writer['ECE'][-1]:.4f}\nReward {writer.data_value_writer['reward'][-1]:.4f}\n")
            update_count+=1
            if not update_count%update_freq:
                ppo_trainer.save_pretrained(f"Agent_weight/PPO_Agent_{writer.determint}_{epoch}_{stats['ppo/loss/total']:.4f}")
                update_count=0
            bar.set_description_str(f"Epoch {epoch}:")

        if epoch in [0,3]:
            input(f"Epoch {epoch}, Press Enter to continue training, You can check the training result now...")

        ppo_trainer.log_stats(stats, instruct_token, Reward,columns_to_log=["query", "response"])
        torch.save(stats,f"{writer.data_folder}/{epoch}_state.pth")

        #### Save model
        # stream.synchronize()
        # dist.barrier()
        # if rank==0:
        Agent_addres=f"Agent_weight/PPO_Agent_{writer.determint}_{epoch}_{stats['ppo/loss/total']:.4f}"
        ppo_trainer.save_pretrained(Agent_addres)
        t.set_description_str(f"Epoch {epoch}:")
    return Agent_addres
        # dist.barrier()
    # cleanup()

def main():

    Training_Config={
        "dataset_path":f'response_result/20240601/din0s_asqa_gpt-3.5-turbo-0125_vanilla_Long_QA.json',
        'deliminator':"06122032_vanilla_f1_r2",
        'max_epoch': 10,
        'trian_batch_size':128,
    }


    # pretrained_model_path=""
    pretrained_model_path=f"Agent_weight/PPO_Agent_{Training_Config['deliminator']}_2_0.0090"

    writer=rl_writer(Training_Config['deliminator'])

    os.makedirs('Agent_weight', exist_ok=True)
    torch.cuda.empty_cache()
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus


    config = PPOConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        # log_with="tensorboard",
        batch_size=Training_Config['trian_batch_size'],
        mini_batch_size=16,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.9,
        init_kl_coef=0.3,
        adap_kl_ctrl=True,
        gradient_accumulation_steps=8,
        ppo_epochs=16,
        is_peft_model=True,
        ratio_threshold= 100.0,
        max_grad_norm=1,
    )

    # set_seed(config.seed)

    # current_device=1
    device_map = {"": Accelerator().local_process_index}
    peft_config = peft.AdaptionPromptConfig(adapter_len = 8, adapter_layers = 24)

    if os.path.isdir(pretrained_model_path):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_path,token=key['hugginface']["token"],torch_dtype=torch.float16,use_cache=True,device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,token=key['hugginface']["token"])
        print("="*50+"Load From Pretrained !!!"+"="*50)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, peft_config = peft_config, token=key['hugginface']["token"],torch_dtype=torch.float16,use_cache=True,device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name,token=key['hugginface']["token"])
        print("="*50+"Load From Huggingface !!!"+"="*50)

    optim_confg=[{
        'params':model.v_head.parameters(),
        'lr':1e-4
    },{
        'params':model.pretrained_model.parameters(),
        'lr':1e-2
    }
                 ]
    ## tokenizer init
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    optim = torch.optim.AdamW(optim_confg, eps = 1e-4,weight_decay=0.02)
    # scheduler1 = StepLR(optim, step_size=9, gamma=0.9)
    scheduler2 = PolynomialLR(optim,  total_iters=270, power=1.5)

    # main_schedualer=SequentialLR(optim, schedulers=[scheduler1, scheduler2], milestones=[54])
    # optim = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        optimizer=optim,
        lr_scheduler=scheduler2,
        )

    # device = 0 if torch.cuda.is_available() else "cpu"
    # ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin


    Dataloader=eval_dataloader(dataset_path=Training_Config['dataset_path'], batch_size=Training_Config['trian_batch_size'], purpose='refine',tokenizer=tokenizer,shuffle=True)

    # model,Dataloader,optim ,lr_scheduler= accelerator.prepare(model,Dataloader,optim,lr_scheduler)

    generation_kwargs = {
        "min_length": -1,
        'temperature': 1,
        # "max_length": 512,
        "max_new_tokens": 128,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        'no_repeat_ngram_size':4
        }


    Agent_addres=trainer(world_size, Training_Config['max_epoch'], ppo_trainer, tokenizer,Dataloader,generation_kwargs,writer)

    # inference(Training_Config['dataset_path'],f"din0s_asqa_{Training_Config['deliminator']}_Vanilla.json",Agent_addres)
    # Show_mean_result(f"din0s_asqa_{Training_Config['deliminator']}_Vanilla.json")

if __name__=="__main__":

    main()

