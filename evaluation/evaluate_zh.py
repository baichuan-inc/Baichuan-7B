import datasets
import os
import torch
import numpy as np
import json
import random
import collections
import sentencepiece as spm
import time
import argparse


def get_time():
    return time.strftime("%Y%m%d %H:%M:%S", time.localtime()) 


OPENMODEL_PATH = 'your/model/path'
if OPENMODEL_PATH == 'your/model/path':
    raise('Model path is not valid')
CEVAL_DATA_PATH = 'your/ceval/data/path'
results_filename = 'results.txt'


def load_model(model_id):
    print('Loading', model_id)
    model_id = os.path.join(OPENMODEL_PATH, model_id)
    if 'gpt2' in model_id:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model = GPT2LMHeadModel.from_pretrained(model_id, device_map='auto',)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    elif 'glm' in model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
    elif 'bloomz' in model_id:
        from transformers import BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = BloomForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    elif 'tigerbot' in model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    elif 'Ziya-LLaMA' in model_id:
        from transformers import AutoTokenizer, LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)     
    elif 'falcon' in model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, 
                                                  trust_remote_code=True, 
                                                  add_bos_token=False,
                                                  add_eos_token=False,
                                                  padding_dise='right')
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)
    elif 'aquila' in model_id:
        from flagai.auto_model.auto_loader import AutoLoader
        from flagai.model.predictor.predictor import Predictor
        from flagai.data.tokenizer import Tokenizer
        import bminf
        model_id = model_id.split('/')[-1]
        print(model_id)
        loader = AutoLoader(
            "lm",
            model_dir=OPENMODEL_PATH,
            model_name=model_id,
            use_cache=True,
            device="cuda")
        model = loader.get_model()
        model.eval()
        model.half()
        tokenizer = loader.get_tokenizer()
    elif 'Open-Llama' in model_id:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True, add_eos_token = False)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)
    candidates = ['A', 'B', 'C', 'D']
    for can in candidates:
        can_idx = tokenizer.encode(can)
        print(can, can_idx)

    return model, tokenizer

class BaseEval:
    def __init__(self, model_id, data_path, show_detail=False):
        self.data_path = data_path
        self.model_id = model_id
        self.task_names = self.get_task_names()
        self.show_detail = show_detail
        self.max_new_tokens = 3  # 最多生成的token数

    def get_task_names(self):
        pass
    
    def set_model_and_tokenizer(self, model, tokenizer):
        self.model = model
        self.tokenizer=tokenizer

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass
        
    def build_prompt(self, task_name, data, dataset, shot):
        pass
    
    def get_results_by_ABCD_prob(self, output, max_new_tokens):
        i = 0  # 用生成的第i个token进行预测
        if 'aquila' in self.model_id:
            scores = output.scores[i].to(torch.float32).detach()
        else:
            scores = output.scores[i][0].to(torch.float32)
        label_score = []
        candidates = ['A', 'B', 'C', 'D']
        for can in candidates:
            can_idx = self.tokenizer.encode(can)[-1]
            label_score.append(scores[can_idx].cpu().numpy())
        answer = candidates[np.argmax(label_score)]

        if self.show_detail:
            # 生成的具体内容
            if 'aquila' in self.model_id:
                outstr = output.text
            else:
                outstr = self.tokenizer.decode(output.sequences.cpu().numpy()[0][-max_new_tokens: ])
            print(f'Generate text (Start>>>){outstr}(<<<End)')
            print(f'Predict answer with ABCD probs: {answer}')
        return answer

    def predict_instance(self, prompt):
        if self.show_detail:
            print(100*'-')
            print(prompt)
        max_new_tokens = self.max_new_tokens
        if 'aquila' not in model_id:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to('cuda')
        if 'glm' in self.model_id:
            # glm 需要特殊处理 https://github.com/mymusise/ChatGLM-Tuning/issues/150
            input_ids[0][-2] = 130001
            input_ids[0][-1] = 130004
        if 'aquila' in model_id:
            from aquila import aquila_generate
            output = aquila_generate(self.tokenizer, self.model, [prompt,], max_gen_len=1)
        else:
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, temperature=0.1, top_p=0.5, repetition_penalty=1.1)
        return self.get_results_by_ABCD_prob(output, max_new_tokens)
        
    def run_a_task(self, task_name, shot, split='test'):
        results = {}
        correct = []
        prompts = collections.OrderedDict()
        print(self.data_path, task_name)
        dataset = datasets.load_dataset(self.data_path, task_name)
        with torch.no_grad():
            for data in dataset[split]:
                prompt = self.build_prompt(task_name, data, dataset, shot)
                plabel = self.predict_instance(prompt)
                idx = data['id']
                results[str(idx)] = plabel
                correct.append(plabel == data['answer'] if 'answer' in data else 0)
                prompts[str(idx)] = prompt
        return {task_name: results}, {task_name: prompts}, {task_name: correct}
    
    def run(self, task_names, shot, split):
        results = dict()
        task_prompts = dict()
        task_acc = dict()
        filename_pfx = f"{self.model_id.split('/')[-1]}_{self.__class__.__name__}_{len(task_names)}_{split}_{shot}-shot"
        for name in task_names:
            print(100*'-')
            print(name)
            value, prompts, acc = self.run_a_task(name, shot, split)
            results.update(value)
            task_prompts.update(prompts)
            task_acc.update(acc)
            json.dump(results, open(f'{filename_pfx}_results.json', 'w'), indent=True, ensure_ascii=False)
        
        outputline = []
        outputline.append(f'{get_time()}\t{filename_pfx}')
        correct = []
        for k, v in task_acc.items():
            correct += v
        outputline.append(f'\tAverage example acc: {np.mean(correct):.4f}')
        acc = []
        for k, v in task_acc.items():
            acc.append(np.mean(v))
        outputline.append(f'\tAverage subject acc: {np.mean(acc):.4f}')
        [print(line) for line in outputline]
        self.save_results(outputline)
    
    def save_results(self, lines):
        with open(results_filename, 'a') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    
    def generate_sample(self, task_name='law', split='validatoin', shot=0):
        dataset = datasets.load_dataset(self.data_path, task_name)
        data = dataset[split][0]
        print('Data' + 100*'-')
        print(data)
        print(f'{shot}-shot prompt' + 100*'-')
        print('(Start>>>)'+self.build_prompt(task_name, data, dataset, shot)+'(<<<End)')

    
class CEval(BaseEval):
    task2desc = {
        "high_school_physics": "高中物理",
        "fire_engineer": "注册消防工程师",
        "computer_network": "计算机网络",
        "advanced_mathematics": "高等数学",
        "logic": "逻辑学",
        "middle_school_physics": "初中物理",
        "clinical_medicine": "临床医学",
        "probability_and_statistics": "概率统计",
        "ideological_and_moral_cultivation": "思想道德修养与法律基础",
        "operating_system": "操作系统",
        "middle_school_mathematics": "初中数学",
        "chinese_language_and_literature": "中国语言文学",
        "electrical_engineer": "注册电气工程师",
        "business_administration": "工商管理",
        "high_school_geography": "高中地理",
        "modern_chinese_history": "近代史纲要",
        "legal_professional": "法律职业资格",
        "middle_school_geography": "初中地理",
        "middle_school_chemistry": "初中化学",
        "high_school_biology": "高中生物",
        "high_school_chemistry": "高中化学",
        "physician": "医师资格",
        "high_school_chinese": "高中语文",
        "tax_accountant": "税务师",
        "high_school_history": "高中历史",
        "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
        "high_school_mathematics": "高中数学",
        "professional_tour_guide": "导游资格",
        "veterinary_medicine": "兽医学",
        "environmental_impact_assessment_engineer": "环境影响评价工程师",
        "basic_medicine": "基础医学",
        "education_science": "教育学",
        "urban_and_rural_planner": "注册城乡规划师",
        "middle_school_biology": "初中生物",
        "plant_protection": "植物保护",
        "middle_school_history": "初中历史",
        "high_school_politics": "高中政治",
        "metrology_engineer": "注册计量师",
        "art_studies": "艺术学",
        "college_economics": "大学经济学",
        "college_chemistry": "大学化学",
        "law": "法学",
        "sports_science": "体育学",
        "civil_servant": "公务员",
        "college_programming": "大学编程",
        "middle_school_politics": "初中政治",
        "teacher_qualification": "教师资格",
        "computer_architecture": "计算机组成",
        "college_physics": "大学物理",
        "discrete_mathematics": "离散数学",
        "marxism": "马克思主义基本原理",
        "accountant": "注册会计师",
    }

    def get_task_names(self):
        task_names = list(sorted(self.task2desc.keys()))
        return task_names
    
    def build_an_example_prompt(self, data, with_answer=True):
        choice =  ['A. '+data["A"],'B. '+data["B"],'C. '+data["C"],'D. '+data["D"]]
        answer = data["answer"].strip().upper() if with_answer else ""
        question = data['question']
        cstr = '\n'.join(choice)
        return f"{question}\n{cstr}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = f"以下是中国关于{self.task2desc[task_name]}考试的单项选择题，请选出其中的正确答案。\n\n{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            shuffle_train = dataset['dev'].shuffle(seed=123)
            prompt = f"以下是中国关于{self.task2desc[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            for i in range(min(len(shuffle_train), shot)):
                prompt += '\n' + self.build_an_example_prompt(shuffle_train[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=str, required=True)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--show_detail', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print(args.model_id, args.shot, args.split, args.task)
    model_id = args.model_id
    shot = args.shot
    split = args.split
    show_detail = args.show_detail

    model, tokenizer = None, None
    if model_id is not None:
        model, tokenizer = load_model(model_id)

    if args.task == 'ceval':
        data_path = CEVAL_DATA_PATH
        if data_path == 'your/ceval/data/path':
            raise('CEval data path is not valid')
        ceval = CEval(model_id, data_path, show_detail=show_detail)
        ceval.set_model_and_tokenizer(model, tokenizer)
        ceval.generate_sample(task_name='law', split=split, shot=3)
        ceval.run(task_names=ceval.task_names, shot=shot, split=split)
    else:
        raise(f'Task is not valid: {args.task}')

