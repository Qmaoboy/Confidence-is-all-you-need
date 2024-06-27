
from LLM_API import *
from qadataset_dataloader import qadataset_dataloader
from rouge_score import rouge_scorer
from prompt_strategy import prompter
from tqdm import tqdm
from util import *
import yaml
import multiprocessing as mp
from sklearn.metrics import roc_auc_score

if os.path.isfile("api_key.yml"):
    with open("api_key.yml","r") as f:
        key=yaml.safe_load(f)


def Load_cot_exmple():
    with open() as f:
        data=f.read().strip()
    print(data)

def question_to_prompt(question,task="self_polish",stretagy='self_polish'):
    p=prompter()
    p.setup_task(task)
    return p.get_prompt(question,[],stretagy)

f
def Get_auroc(accuracy,confidence_scores):
    y_true=np.where(np.array(accuracy) < 0.3,0,1)
    return roc_auc_score(np.array(y_true), np.array(confidence_scores))

def get_from_gpt(prompt,parser_stretagy):
    result=GPT_API('gpt-4-turbo',key['openai']['api_key'],parser_stretagy,prompt).generate()
    return result

def get_from_calude(prompt,parser_stretagy):
    result=GPT_API('claude-3-5-sonnet-20240620',key['openai']['api_key'],parser_stretagy,prompt).generate()
    return result

def ans_scorer(new_ans,original_ans):
    ## Compare Result
    result=rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(new_ans,original_ans)['rougeL'].fmeasure
    return result

def rewrite_worker(share_list,idx,original_question,ground_truth,documnet,baseline):

    if baseline =="vanilla":
        Answer_result=get_from_gpt(question_to_prompt([original_question],'pure','vanilla'),'confidence')
        if Answer_result is not None:
            rouge_score=ans_scorer(Answer_result['Answer'],ground_truth)
            share_list.append({
                'Id':idx,
                'Original_question':original_question,
                'Documnet':documnet,
                'Ground_truth':ground_truth,
                'Answer':Answer_result['Answer'],
                'Confidence':Answer_result['Confidence'],
                'rouge_score':rouge_score,
            })

    elif baseline =="self_polish":
        ###### self polish iterate refine question
        old_refine_question=original_question
        for _ in (p:=tqdm(range(3),leave=False)):
            new_question=get_from_gpt(question_to_prompt([old_refine_question],'self_polish','self_polish'),'self_polish')
            if old_refine_question==new_question['New_Question']: ## converge
                break
            else:
                old_refine_question=new_question['New_Question']

        Answer_result=get_from_gpt(question_to_prompt([new_question["New_Question"]],'QA','vanilla'),'confidence')
        if Answer_result is not None:
            rouge_score=ans_scorer(Answer_result['Answer'],ground_truth)
            share_list.append({
                'Id':idx,
                'Original_question':original_question,
                'New_Question':new_question['New_Question'],
                'Documnet':documnet,
                'Ground_truth':ground_truth,
                'Answer':Answer_result['Answer'],
                'Confidence':Answer_result['Confidence'],
                'rouge_score':rouge_score,
            })

    elif baseline =="RaR":
        Answer_result=get_from_gpt(question_to_prompt([original_question],'RaR','RaR'),'RaR')
        # Answer_result=get_from_gpt(question_to_prompt([new_question],'RaR','RaR'),'confidence')
        if Answer_result is not None:
            rouge_score=ans_scorer(Answer_result['Answer'],ground_truth)

            share_list.append({
                'Id':idx,
                'Original_question':original_question,
                'Expanded_question':Answer_result['Expanded_Question'],
                'Answer':Answer_result['Answer'],
                'Ground_truth':ground_truth,
                'Confidence':Answer_result['Confidence'],
                'rouge_score':rouge_score,
                'Documnet':documnet,
            })

def evaluate_result(baseline):
    with open(f'{baseline}.json','r') as f:
        data=json.load(f)

    acc=np.array([float(i['rouge_score']) for i in data])
    conf=np.array([float(i['Confidence']) for i in data])
    ece_score=np.abs(acc-conf)
    print(baseline)
    print(f"rouge_score mean :{np.mean(acc)}")
    print(f"ECE mean :{np.mean(ece_score)}")
    print(f"Auroc mean :{Get_auroc(acc,conf)}")

def main(baseline):

    train_dataloader=qadataset_dataloader(dataset_path="din0s/asqa",split='dev',batch_size=1).trainloader
    share_list=mp.Manager().list()
    mp_pool=mp.Pool(processes=mp.cpu_count())
    for idx,(batch,Ground_truth) in enumerate(progress:=tqdm(train_dataloader)):
        original_question=[i[0] for i in batch]
        document=[search_wikipedia_byurl(i[1]) if i[2] else i[1] for i in batch]
        args=[(share_list,idx,q,gt,doc,baseline) for idx,(q,gt,doc) in enumerate(zip(original_question,Ground_truth,document))]
        mp_pool.starmap(rewrite_worker,args)
        if share_list:
            progress.set_description_str(f"Processing {len(share_list)} batch rouge {np.mean(np.array([i['rouge_score'] for i in share_list]))}")
        if len(share_list)>=50:
            break

    mp_pool.close()
    mp_pool.join()

    with open(f'{baseline}.json','w+') as f:
        json.dump(list(share_list),f)


if __name__=="__main__":
    baseline='vanilla'
    main(baseline)
    evaluate_result(baseline)
