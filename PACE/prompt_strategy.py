from util import setup_logger
import torch
import numpy as np
from datetime import datetime
# activation_time=datetime.now().strftime("%Y%m%d")
logger = setup_logger(f'log/response_{datetime.now().strftime("%Y%m%d")}.log')

class prompter:
    def __init__(self):
        self.setup_assis_prompt()

    def setup_assis_prompt(self):

        self.confidence_define_prompt="Note: The confidence indicates how likely you think your Answer is true and correct,from 0.00 (worst) to 1.00 (best)"

        self.similarity_prompt="Note: The similarity indicates how likely you think your Answer and document is semantic related,from 0.00 (worst) to 1.00 (best)"

        self.acc_prompt="Note: The accuracy indicates how likely you think your Answer and document is semantic accuracy,from 0.00 (worst) to 1.00 (best)"

    def setup_task(self,task):
        if task:
            if task=="QA":
                self.answer_type="provide Answer to the question and confidence to the Answer"
                self.system_prompt=f"This is a QA task, please {self.answer_type} in json."

            elif task=="Long_QA":
                # self.answer_type="provide very long Answer with more details to the question and confidence to the Answer"
                self.answer_type="provide Answer to the question and confidence to the Answer"
                self.system_prompt=f"This is a QA task, please {self.answer_type} in json."

            elif task=="similarity":
                self.system_prompt=f"This is a similarity compare task, please {self.answer_type} in json."

        else:
            raise ValueError("task Not Recognized")

    def get_prompt(self,query:list,document:list,stretagy:str):

        # if with_rag:
        #     # logger.info(f"{stretagy} activate knwledge {with_rag}")
        #     doc_str=" ".join(document)
        #     self.rag_inject=f"base on the given Knowledge"
        #     self.document=f"Knwoledge : {doc_str},\n"
        if stretagy=="vanilla":
            return self.vanilla_prompt(query,document)
        elif stretagy=="cot":
            return self.chain_of_thought(query,document)

        elif stretagy=="multi_step":
            return self.multi_step(query,document)

        elif stretagy=="similarity":
            return self.document_answer_similarity(query,document)
        # elif stretagy=="acc":
        #     return self.answer_acc(query,document)


    def document_answer_similarity(self,answer:list,document:list)-> dict:
        # logger.info(f"{answer} {type(document)}")
        document_str="\n".join(document)
        answer="".join(answer)
        Instruction=f"Compare the semantic similarity between given groudtruth and Answer"
        input_text=f"groudtruth:{document_str},\nAnswer:{answer},\n\nresponse format :\nsimilarity:[Your final similarity here]"
        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":"",'input_text':input_text,"assit_prompt":self.similarity_prompt}

    def vanilla_prompt(self,question:list,document:list)-> dict:
        question=question.pop()

        Instruction=f"Now, Read the Question and {self.answer_type}\n"
        vanilla_prompt=f'''\nOnly give me the reply according to response format, don't give me any other words.\n\nresponse format :\nAnswer: [Your final Answer here],\nConfidence : [Your final Confidence here]\n'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"{question}",'input_text':vanilla_prompt,"assit_prompt":self.confidence_define_prompt}

    def chain_of_thought(self,question:list,document:list)-> dict:
        question=question.pop()

        Instruction=f"Now, Read the Question and Let's think it step by step.{self.answer_type} and give the Explanation to the Answer\n"

        cot_prompt = f'''\nOnly give me the reply according to response format, don't give me any other words.\n\nresponse format:\nAnswer:[Your Answer here],\nConfidence:[Your Confidence here],\nExplanation:[Your final Explanation here]\n'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"{question}",'input_text':cot_prompt,"assit_prompt":self.confidence_define_prompt}

    def multi_step(self,question:list,document:list)->dict:
        question=question.pop()
        step_prompt=f"Step 1: [Your reasoning]... Step k : [Your reasoning]"

        Instruction=f"Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then {self.answer_type}\n"

        multi_step_prompt = f'''\nOnly give me the reply according to response format, don't give me any other words.\n\nresponse format:\n{step_prompt}\nAnswer:[ONLY Your Final Answer here],\nConfidence:[Your Overall Confidence here]'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"{question}",'input_text':multi_step_prompt,"assit_prompt":self.confidence_define_prompt}




