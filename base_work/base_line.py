
from LLM_API import *
from qadataset_dataloader import qadataset_dataloader
from rouge_score import rouge_scorer
from prompt_strategy import prompter
from tqdm import tqdm
from util import *
import yaml,os,json,re
import multiprocessing as mp
from sklearn.metrics import roc_auc_score
from copy import copy,deepcopy
import textgrad as tg

if os.path.isfile("api_key.yml"):
    with open("api_key.yml","r") as f:
        key=yaml.safe_load(f)

 ## gpt-3.5-turbo-0125, gpt-4-turbo
## "claude-3-5-sonnet-20240620"

# api_model='claude-3-5-sonnet-20240620'
# api_key=key['claude']['api_key']

api_model='gpt-4-turbo'
api_key=key['openai']['api_key']

def question_to_prompt(question,task="self_polish",stretagy='self_polish'):
    p=prompter()
    p.setup_task(task)
    return p.get_prompt(question,[],stretagy)

def Get_auroc(accuracy,confidence_scores):
    y_true=np.where(np.array(accuracy) < 0.3,0,1)
    return roc_auc_score(np.array(y_true), np.array(confidence_scores))

def ans_scorer(new_ans,original_ans):
    ## Compare Result
    result=rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True).score(new_ans,original_ans)['rougeL'].fmeasure
    return result

def rewrite_worker(share_list,idx,original_question,ground_truth,documnet,baseline):

    if baseline =="vanilla":
        prompt=question_to_prompt([original_question],'pure','vanilla')
        Answer_result=GPT_API(api_model,api_key,'confidence',prompt).generate()
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
        old_refine_question=deepcopy(original_question)
        for _ in (p:=tqdm(range(3),leave=False)):
            prompt=question_to_prompt([old_refine_question],'self_polish','self_polish')
            new_question=GPT_API(api_model,api_key,'self_polish',prompt).generate()

            if old_refine_question==new_question['New_Question']: ## converge
                break
            else:
                old_refine_question=new_question['New_Question']

        new_question_prompt=question_to_prompt([new_question["New_Question"]],'Long_QA','vanilla')
        Answer_result=GPT_API(api_model,api_key,'confidence',new_question_prompt).generate()

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
        prompt=question_to_prompt([old_refine_question],'RaR','RaR')
        Answer_result=GPT_API(api_model,api_key,'RaR',new_question_prompt).generate()

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
    elif baseline =="textgrad":
        os.environ['OPENAI_API_KEY'] = api_key
        def parser(text):
            answer_match = re.search(r'"answer": "(.*?)"', text)
            if answer_match:
                answer = answer_match.group(1)
            # Regex to capture the confidence score
            confidence_match = re.search(r'"confidence_score": (\d+\.\d+)', text)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))

            return {"Answer":answer,"Confidence":confidence_score}

        tg.set_backward_engine("gpt-3.5-turbo-0125", override=True)

        # Step 1: Get an initial response from an LLM.
        model = tg.BlackboxLLM("gpt-4o")
        question_string = ("who is brack obama?"
                        "provide confidence score to the answer in json")

        question = tg.Variable(question_string,
                            role_description="question to the LLM",
                            requires_grad=False)

        answer = model(question)
        print(answer)
        result=parser(str(answer))
        answer.set_role_description("concise and accurate answer to the question")

        # Step 2: Define the loss function and the optimizer, just like in PyTorch!
        # Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
        # that works with "textual gradients".
        optimizer = tg.TGD(parameters=[answer])
        evaluation_instruction = (f"Here's a question: {question_string}. "
                                "Evaluate any given answer to this question, "
                                "be smart, logical, and very critical. "
                                "Just provide concise feedback."
                                )


        # TextLoss is a natural-language specified loss function that describes
        # how we want to evaluate the reasoning.
        loss_fn = tg.TextLoss(evaluation_instruction)

        # Step 3: Do the loss computation, backward pass, and update the punchline.
        # Exact same syntax as PyTorch!
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        result['Answer']=str(answer)



def evaluate_result(baseline):
    with open(f'{api_model}_{baseline}.json','r') as f:
        data=json.load(f)

    acc=np.array([float(i['rouge_score']) for i in data])
    conf=np.array([float(i['Confidence']) for i in data])
    ece_score=np.abs(acc-conf)
    print(f"{api_model}:{baseline}")
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

    with open(f'{api_model}_{baseline}.json','w+') as f:
        json.dump(list(share_list),f)


if __name__=="__main__":
    baseline='RaR'
    main(baseline)
    evaluate_result(baseline)
