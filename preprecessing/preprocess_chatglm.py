
import json
import transformers
def covert_ids_json(path, save_path, max_seq_length=512, ds_type='train'):
    json_context = []
    if ds_type !='train':
        print('eval or test is not need covert!')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '../chatglm2', trust_remote_code=True)

    config = transformers.AutoConfig.from_pretrained(
        '../chatglm2', trust_remote_code=True, device_map='auto')

    jw = open(save_path, 'w', encoding='utf-8')

    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    for record in data:
        context = record['context']
        target = record['target']

        context_ids = tokenizer.encode(
            context,
            max_length=max_seq_length,
            truncation=True)

        target_ids = tokenizer.encode(
            target,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False)

        input_ids = context_ids + target_ids + [config.eos_token_id]
        samples = {"input_ids": input_ids, "context_len": len(context_ids), 'target_len': len(target_ids), 'ds_type': ds_type}
        json_context.append(samples)

    jw.write(json.dumps(json_context, indent=4, ensure_ascii=False))

def load_eval_json(path):
    with open(path, 'r', encoding='utf-8') as fr:
        context = json.load(fr)

    return context

#
#path = '../preprecessing/stepwise_inference_train.json'
#covert_ids_json(path, save_path='../data/train_imcs_ner_one2one.json')
