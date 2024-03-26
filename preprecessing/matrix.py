# Generated NER matrix
import re

def eval_matrix(pred, target):
    """
    pred: a predication result with text format. eg: "疾病:胸膜疾病、小儿胸膜疾病、胸膜炎。"
    target:  a target result with text format. eg: "疾病:胸膜疾病、小儿胸膜疾病、胸膜炎、肺部感染。"
    """
    labels = []
    predications = []

    label = target.split('\n')

    for l in label:

        label_ner_type, label_context_list = l.split(':')
        label_context_list = label_context_list.split('、')
        for context in label_context_list:
            if context.endswith("。"):
                context = context[:-1]
            labels.append((label_ner_type, context))

    results = pred.split('\n')
    for res in results:

        pred_ner_type, pred_context_list = res.split(':')
        if '无'+pred_ner_type in res:
            continue

        pred_context_list = pred_context_list.split('、')
        for context in pred_context_list:
            if context.endswith("。"):
                context = context[:-1]
            predications.append((pred_ner_type, context))

    X, Y, Z = 1e-10, 1e-10, 1e-10

    R = set(predications)
    T = set(labels)
    X += len(R & T)
    Y += len(R)
    Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    return f1, precision, recall


def test_formation(sentence, answer):

    results = answer.split('\n')
    labels = []

    exp = {'bod': '身体组成', 'dis': '疾病', 'ite': '医学检验项目', 'dep': '科室',
           'sym': '临床表现', 'equ': '医疗设备', 'pro': '医疗程序', 'dru': '药物',
           'mic': '微生物类'}
    new_exp = {exp[k]:k for k in exp}

    for l in results:
        l = l.strip()

        type_entity = l.split(':')

        type_name = type_entity[0]
        try:
            entity = type_entity[1].split('?')
        except:
            entity = []
        for e in entity:
            if e.endswith("?"):
                e = e[:-1]
            print(e, sentence)
            e = re.escape(e)
            matches = re.findall(e, sentence)

            for match in matches:
                start = sentence.find(match)
                end = start + len(match)
                if {"start_idx": start, "end_idx": end, "type": new_exp[type_name], "entity": e} not in labels:
                    labels.append({"start_idx": start, "end_idx": end, "type": new_exp[type_name], "entity": e})

    return labels

# input1 = "第十章胸膜疾病小儿胸膜疾病以胸膜炎最为常见，多继发于肺部感染，原发性或其他原因所致者较少见。"
# input2 = "疾病:胸膜疾病、小儿胸膜疾病、胸膜炎、肺部感染。"

# pred = "疾病:胸膜疾病、小儿胸膜疾病、胸膜炎。"
# label = "疾病:胸膜疾病、小儿胸膜疾病、胸膜炎、肺部感染。"
# f1, precision, recall = eval_matrix(pred, label)
# print(f1)

#print(find_all_positions(input1, input2))
