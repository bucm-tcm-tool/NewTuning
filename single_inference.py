from torch.utils.data import DataLoader
from dataloaders.data_helper_ import TextLoader,collate_fn
from dataloaders.preprocess_chatglm import load_eval_json
from transformers.trainer import get_scheduler
from accelerate import notebook_launcher
from accelerate import Accelerator
import torch
from adapter_model import AdapterModel
from tqdm import tqdm
from utils.matrix import test_formation
import math
import os
import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging,sys
import datetime
def test():
    jw = open('/root/CMeEE_adapter_research/code/results/adapter_series_same_weight_FFN_Single_Inference.json', 'w', encoding='utf-8')
    adapter_path = '/root/CMeEE_adapter_research/code/adapter_save_path/adapter_series_same_weight_FFN_Single_Inference.pth'
    model_path = '/root/CMeEE_adapter_research/code/chatglm2'
    test_json_dir = '/root/CMeEE_adapter_research/code/data/CMeEE-V2/CMeEE-V2_test.json'

    # --------------loading dataset------------------

    # build dataloader
    test_eval = load_eval_json(test_json_dir)

    # -------------loading model---------------------
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).cuda()
    # ------------loading adatper model--------------
    if adapter_path:
        model.load_state_dict(torch.load(adapter_path), strict=False)

    text = '下列句子中可能存在涉及身体组成、疾病、医学检验项目、科室、临床表现、医疗设备、医疗程序、药物、微生物类等类型的文字描述，请按“涉及的类型:具体的文字内容”的格式输出，多个内容之间以顿号间隔，每输出完一种类型涉及的内容后，换行继续输出下一类型与对应的文字内容，直到所有类型输出完毕，如：疾病:心律失常。\n身体组成:头部、躯干、四肢。\n句子：'
    model.eval()
    num = 0
    all_results = []
    for sample in test_eval:


        context = sample['text'].replace('、', '，')
        context = text+context
        print('question:{}'.format(context))
        starttime = datetime.datetime.now()
        response, history = model.chat(tokenizer, context, history=[])
        print('response:{}'.format(response))
        endtime = datetime.datetime.now()
        cum_time = (endtime -starttime).seconds * 1000 + (endtime -starttime).microseconds / 1000
        all_results.append({'text': sample['text'], 'entities': response, 'cum_time':cum_time})

    jw.write(json.dumps(all_results, indent=4, ensure_ascii=False))
test()