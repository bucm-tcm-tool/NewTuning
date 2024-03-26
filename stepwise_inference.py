from torch.utils.data import DataLoader
from dataloaders.data_helper_ import TextLoader,collate_fn
from dataloaders.preprocess_chatglm import load_eval_json
from transformers.trainer import get_scheduler
from accelerate import notebook_launcher
from accelerate import Accelerator
import torch
from adapter_model import AdapterModel
from tqdm import tqdm
from utils.matrix import eval_matrix
import math
import os
import datetime
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging,sys
import json
def test():
    jw = open('/root/CMeEE_adapter_research/code/results/adapter_series_same_weight_FFN_Stepwise_Inference.json', 'w', encoding='utf-8')
    model_path = '/root/CMeEE_adapter_research/code/chatglm2'

    test_json_dir = '/root/CMeEE_adapter_research/code/data/CMeEE-V2/CMeEE-V2_test.json'
    adapter_path = '/root/CMeEE_adapter_research/code/adapter_save_path/adapter_series_same_weight_FFN_Stepwise_Inference.pth'


    # --------------loading dataset------------------

    # build dataloader

    dl_eval = load_eval_json(dev_json_dir)

    # -------------loading model---------------------
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).cuda()

    # ------------loading adatper model--------------
    if adapter_path:
        model.load_state_dict(torch.load(adapter_path), strict=False)

    model.eval()
    choose = "对于身体组成、疾病、医学检验项目、科室、临床表现、医疗设备、医疗程序、药物、微生物类等类型，下列句子包含哪几类？请直接输出涉及的类型，若涉及多个类型则以顿号间隔，如：身体组成、药物、微生物类。\n句子："
    question = "下列句子中涉及{}的文字内容有哪些？请按”{}:涉及的具体内容”的格式输出，涉及多个具体内容时，以顿号间隔。\n句子："
    all_result = []
    for sample in dl_eval:
        text = sample['text']
        context = choose+text.replace('、', '，')

        print('question:{}'.format(context))
        starttime = datetime.datetime.now()
        response, history = model.chat(tokenizer, context, history=[])
        print('response:{}'.format(response))
        answer = response.split('、')

        query_answers = []
        if len(answer) > 50:
            endtime = datetime.datetime.now()
            cum_time = (endtime - starttime).seconds * 1000 + (endtime - starttime).microseconds / 1000
            all_result.append({'text': sample['text'], 'entities': query_answers, 'cum_time': cum_time})
            continue
        for a in answer:
            if a.endswith("。"):
                a = a[:-1]
            if len(a) > 50:
                continue
            query = question.format(a,a)+text.replace('、', '，')

            response, history = model.chat(tokenizer, query, history=[])
            print('--response:{}'.format(response))
            if response !='':
                query_answers.append(response)

        endtime = datetime.datetime.now()
        cum_time = (endtime -starttime).seconds * 1000 + (endtime -starttime).microseconds / 1000

        print(cum_time)
        response = '\n'.join(query_answers)
        all_result.append({'text':sample['text'], 'entities':query_answers,'cum_time':cum_time})

    jw.write(json.dumps(all_result, indent=4, ensure_ascii=False))
test()
