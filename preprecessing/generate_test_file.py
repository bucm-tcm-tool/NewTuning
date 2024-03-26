
from preprocess_chatglm import load_eval_json
import os
import json


def generate(json_dir):
    save_path = json_dir.split('/')

    save_path = save_path[3]+'/CMeEE-V2_test.json'
    jw = open(save_path, 'w', encoding='utf-8')

    dl_test = load_eval_json(json_dir)
    dl_test2 = load_eval_json('/root/CMeEE_adapter_research/code/data/CMeEE-V2/CMeEE-V2_test.json')
    all_results = []
    total_time = 0
    for sample,sample2 in zip(dl_test,dl_test2):

        if sample['text'] != sample2['text']:
            sample['text'] = sample2['text']

        label = test_formation(sample['text'], sample['entities'])
        if label == []:
            label =[{"start_idx": 0, "end_idx": 0, "type": "dis", "entity": ""}]
        all_results.append({'text': sample['text'], 'entities': label})
    print(total_time)
    jw.write(json.dumps(all_results, indent=4, ensure_ascii=False))


dev_json_dir = '/root/CMeEE_adapter_research/code/results/adapter_series_same_weight_FFN_Single_Inference.json'
generate(dev_json_dir)
