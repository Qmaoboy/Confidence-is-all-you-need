
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from util import *
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from RL_env import Environment,vanilla_prompt,reward_function

def trianer():


    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Define RL training parameters
    num_episodes = 1000
    learning_rate = 5e-5

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    ## init Vanilla Prompt p_0
    Instruction="Now,Read the Question and provide Answer to the question and confidence to the Answer."

    ## Load Dataloader
    data_loader=eval_dataloader(dataset_path="response_result/20240601/natural_questions_gpt-3.5-turbo-0125_vanilla_QA_Cos_sim_No_Shuffle_bertscore.json",batch_size=2,purpose='refine').loader

    print("start to train")
    for prompt,ans,ground_Truth,Confidence,Document in data_loader:
        Instruction=[i['Instruction'] for i in prompt]
        question=[i['Question'] for i in prompt]
        print(Instruction)
        # Training loop
        for episode in range(num_episodes):
            # Sample a prompt from the dataset (not implemented here, you need to replace this)

            # Tokenize the prompt
            tokens= tokenizer(Instruction, return_tensors="pt",padding=True,truncation=True)
            input_ids = tokens.input_ids
            attention_mask= tokens.attention_mask
            # attention_mask = torch.ones_like(input_ids)

            # Generate prompt continuation using GPT-2

            batch_outputs = model.generate(input_ids, attention_mask=attention_mask,max_length=100, do_sample=True, temperature=0.8, top_p=0.9, top_k=50, num_return_sequences=1)

            generated_instruction = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
            print(Instruction)
            print("*"*50)
            print(generated_instruction)
            prompt=list(map(vanilla_prompt,question,generated_instruction))
            Answer_batch=[]
            conf_batch=[]
            for p in (api:=tqdm(prompt)):
                conf,answer=Environment(p,'gpt-3.5-turbo-0125')
                Answer_batch.append(answer)
                conf_batch.append(conf)
                api.set_description_str(f"Confidence:{conf},Answer:{answer}")
            # Evaluate the quality of the generated prompt (not implemented here, you need to replace this)
            reward = reward_function([answer],[ground_Truth],[conf_batch],Document)
            print(reward)
            # Compute loss (negative reward in this case)
            # loss = loss * reward
            # reward_tensor = torch.tensor(reward, requires_grad=True)
            loss = -reward

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print episode information
            print(f"Episode {episode+1}/{num_episodes}, Loss: {loss.item()}, Reward: {reward}")


if __name__=="__main__":
    trianer()
