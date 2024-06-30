from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
from datetime import datetime
from util import *
import os,re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
logger = setup_logger(f'log/response_{datetime.now().strftime("%Y%m%d")}.log')

class qadataset_dataloader:
    def __init__(self,dataset_path="din0s/asqa",split='train',batch_size=2,shuffle=True):
        super(qadataset_dataloader,self).__init__()
        self.dataset_path=dataset_path
        self.split=split
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.get_loader()


    def setup_collect_fn(self): ## "gsm8k",DateUnd,Prf-Law,Biz-Ethics
        if self.dataset_path=="din0s/asqa":
            dataset_name="din0s/asqa"
            self.collect_fn=self.asqa_collate_fn
            self.dataset = load_dataset(dataset_name)

        elif self.dataset_path=="natural_questions": # Short Ans ; EM Score
            self.collect_fn=self.naturalqa_collect_fn
            self.dataset=self.Load_natural_qa(20)

        elif self.dataset_path=="ChilleD/StrategyQA": # True/ False
            dataset_name="ChilleD/StrategyQA"
            self.collect_fn=self.stretagy_qa_collect_fn
            self.dataset = load_dataset(self.dataset_path)

        elif self.dataset_path=="gsm8k":
            dataset_name="gsm8k"
            self.collect_fn=self.gsm8k_collect_fn
            self.dataset = load_dataset(self.dataset_path,"main")

        elif self.dataset_path=="triviaQA":
            dataset_name="mandarjoshi/trivia_qa"
            self.collect_fn=self.triviaqa_collect_fn
            self.dataset = load_dataset(dataset_name,"rc.wikipedia")

    def get_loader(self):

        self.setup_collect_fn()

        if self.split in self.dataset:
            self.trainloader=DataLoader(self.dataset[self.split],batch_size=self.batch_size,shuffle=self.shuffle,collate_fn=self.collect_fn,num_workers=1,drop_last=True)
        else:
            self.trainloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collect_fn,shuffle=self.shuffle)

    def Load_natural_qa(self,idx):
        qa_dict=[]
        for i in range(idx):
            datapath=f"natural_qa_dataet/natual_qa_{i}.json"
            old_data=load_checkpoint(datapath)
            qa_dict+=old_data
            logger.info(f"Load Data from {datapath}")
        return qa_dict

    def asqa_collate_fn(self,batch): ## long_ans

        isurl=True
        res=[[i['ambiguous_question'],[i['wikipages'][0]['url']],isurl] for i in batch]
        long_ans=[i['annotations'][0]['long_answer'] for i in batch]
        return res,long_ans

    def naturalqa_collect_fn(self,batch): ## Long ans or short ans
        res=[[i['Question'],i['Document'],i['isurl']] for i in batch]
        short_ans=["".join(i['Short_answer']) for i in batch]
        long_ans=["".join(i['long_answer'])for i in batch]
        return res,short_ans

    def stretagy_qa_collect_fn(self,batch): ## True / False
        res=[]
        isurl=False
        res=[[i['question'],i['facts'],isurl]for i in batch]
        answer=[i['answer'] for i in batch]
        return res,answer

    def gsm8k_collect_fn(self,batch):

        res=[[i["question"],"",False] for i in batch]
        ans=[i['answer']for i in batch]
        ground_Truth=[re.findall(r'\d+',i['answer']).pop()for i in batch]

        return res,ans

    def triviaqa_collect_fn(self,batch):
        res=[]
        print(batch)
        return batch

class eval_dataloader:
    def __init__(self,dataset_path,batch_size,purpose='compare',tokenizer="",shuffle=False) -> None:
        self.tokenizer = tokenizer
        self.shuffle=shuffle
        self.dataset = load_checkpoint(dataset_path)
        if purpose=="compare":
            self.loader=DataLoader(list(self.dataset),batch_size=batch_size,collate_fn=self.parallel_simi_collate_fn,shuffle=self.shuffle,num_workers=1,drop_last=True,persistent_workers=True)
        elif purpose=="acc":
            self.loader=DataLoader(list(self.dataset),batch_size=batch_size,collate_fn=self.acc_collate_fn,shuffle=self.shuffle,num_workers=1,drop_last=True,persistent_workers=True)

        elif purpose=="refine":
            sep_index=int(len(self.dataset)*0.8)
            # sep_index=10
            self.trainloader=DataLoader(list(self.dataset)[:sep_index],batch_size=batch_size,collate_fn=self.refine_collect_fn,shuffle=self.shuffle,num_workers=1,drop_last=True,persistent_workers=True)
            self.testloader=DataLoader(list(self.dataset)[sep_index:],batch_size=batch_size,collate_fn=self.refine_collect_fn,shuffle=self.shuffle,num_workers=1,drop_last=True,persistent_workers=True)

    def acc_collate_fn(self,batch):
        # print(batch)
        ans=[i['Answer'] for i in batch]
        ground_Truth=[i['Ground_Truth']for i in batch]
        res=[[i['Document'],i['Doc_Ans_simi'],i['Confidence'],i['Prompt']]for i in batch]
        return res,ans,ground_Truth

    def parallel_simi_collate_fn(self,batch):
        ans=[i['Answer'] for i in batch]
        long_ans=[i['Ground_Truth']for i in batch]
        res=[[i['Question'],i['Document'],i['Confidence'],i['Explanation'],i['Prompt']] for i in batch]
        return res,ans,long_ans

    def refine_collect_fn(self,batch):

        ans=[i['Answer'] for i in batch]
        ground_Truth=[i['Ground_Truth']for i in batch]

        Document=[i['Document']for i in batch]
        Confidence=[i['Confidence']for i in batch]
        prompt=[i['Prompt'] for i in batch]

        Instruct_Example="{Instruction:[Your Instruction Here]}"
        basic_instruct="generate Instruction for the given and Question and Instruction"

        instruc_gpt_='Below is an instruction that describes a task. Write a better Instruction base on the Instruction and Question.'

        # instruct=[f'''<s>[INST] <<SYS>>You are a Instruction generator excel at rewriting the basic_instruction according to the question in order to generate more detail Answer for long form QA task. Only Give me the new_Instruction, do not give any other infomation.<</SYS>>\nbasic_instruction:"{i['Prompt']['Instruction']}"\nQuestion:{i['Prompt']['Question']}\n
        # [/INST]new_Instruction:\n''' for i in batch]

        instruct=[f'''[INST] <<SYS>>
                    Rewrite the following basic instruction to help a large language model generate a more detailed and comprehensive answer for a long-form QA task. Ensure the rewritten instruction is clear and concise, prompting the model to provide a thorough and well-structured response of at least 300 tokens to the given question. The new instruction should be within 256 tokens.

                    Basic Instruction: "{i['Prompt']['Instruction']}"
                    Question: "{i['Prompt']['Question']}"
                    [/INST]
                    new instruction:''' for i in batch]


        # gpt2_instruct=[f"{basic_instruct}\n\n \#\#\#Instruction:\n {i['Prompt']['Instruction']},Question:{i['Prompt']['Question']}\,Now Start to generate better Instruction :\n" for i in batch]

        ground_Truth_token=self.tokenizer(ground_Truth,padding=False,truncation=True,max_length=512)
        instruct_token=self.tokenizer(instruct,padding=False,truncation=True,max_length=512)

        return prompt,instruct,instruct_token,ans,ground_Truth,ground_Truth_token,Confidence,Document


if __name__=="__main__":
    qa_loader=qadataset_dataloader("din0s/asqa",split='dev',batch_size=1).loader
    for batch,short_ans in qa_loader:

        break

    # simi_loader=eval_dataloader("response_result/ChilleD_StrategyQA_gpt-3.5-turbo-0125_vanilla_QA_2024_05_12.json",1,'compare').loader
    # for res,ans,long_ans in simi_loader:
    #     print(ans)

