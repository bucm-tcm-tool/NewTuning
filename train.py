from torch.utils.data import DataLoader
from preprecessing.data_helper import TextLoader,collate_fn
from preprecessing.preprocess_chatglm import load_eval_json

from transformers.trainer import get_scheduler
from accelerate import notebook_launcher
from accelerate import Accelerator
import torch
from adapter_model import AdapterModel
from tqdm import tqdm
import math
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging, sys,random
import torch.backends.cudnn as cudnn
import numpy as np
seed=12345
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # zl

def train():

    model_path = '/root/Documents/mChatAdapter/chatglm2'
    train_json_dir = './data/single_inference_train_ids.json' 
    logging_path = '/root/Documents/mChatAdapter/logs'
    batch_size = 1 # Accelerator will use gradient accumulation
    max_epoch = 100
    lr = 1e-4

    logging.basicConfig(filename=logging_path+'/adapter_series_same_weight_FFN_Single_Inference.txt', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    accelerator = Accelerator(gradient_accumulation_steps=4,mixed_precision="bf16")
    # --------------loading dataset------------------
    ds_train = TextLoader(train_json_dir)

    # build dataloader
    dl_train = DataLoader(ds_train, num_workers=0, batch_size=batch_size, pin_memory=True,  generator=torch.Generator().manual_seed(seed), shuffle=True, collate_fn=collate_fn,
                          drop_last=False)

    max_steps = len(dl_train)/batch_size * max_epoch
    print('Total sample:', max_steps)

    # -------------loading model---------------------
    print('loading weights')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).cuda()
    
    # ------------loading adatper model--------------
    print('setting weights')
    adapter_model = AdapterModel()
    adapter_model.set_adapter_weights(model)  # set trainable adapter weights

    # ------------model for gradient_checkpointing--------------
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    opt_param = []

    for param in model.parameters():
        if param.requires_grad:
            opt_param.append(param)

    print('Training param:')
    for name, param in model.named_parameters():

        if param.requires_grad:
            print(name)

    # ------------Training param--------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

    device = accelerator.device

    model, optimizer, scheduler,dl_train = accelerator.prepare(model, optimizer,scheduler,dl_train)

    iter_num = 0
    epoch_steps = len(dl_train)
    print('each epoch is', epoch_steps)
    # ------------Training start--------------
    model.train()
    loss_ = 0
    for epoch_num in tqdm(range(max_epoch)):

        for batch in dl_train:
            with accelerator.accumulate(model):

                context = batch['input_ids'].to(device)
                target = batch['labels'].to(device)

                outputs = model(context, labels=target)
                loss = outputs.loss
                accelerator.backward(loss)
                # loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                #lr_scheduler.step()

            iter_num += 1
            loss_+=loss.item()
            if iter_num % 4 ==0:
                logging.info('iter:%d, loss:%f' % (iter_num, loss_/4))
                loss_=0

            if iter_num*batch_size % 5000 == 0:
                save_param = {}
                all_param = model.state_dict()
                for name_param in all_param:
                    if 'adapter' in name_param:
                        save_param.update({name_param: all_param[name_param]})
                    elif 'prefix' in name_param:
                        save_param.update({name_param: all_param[name_param]})
                torch.save(save_param, './adapter_save_path/adapter_series_same_weight_FFN_Single_Inference_{}.pth'.format(iter_num*batch_size))
                print('model save')
        print('epoch:', epoch_num+1)
        
        save_param = {}
        all_param = model.state_dict()
        for name_param in all_param:
            if 'adapter' in name_param:
                save_param.update({name_param: all_param[name_param]})
            elif 'prefix' in name_param:
                save_param.update({name_param: all_param[name_param]})

        torch.save(save_param, './adapter_save_path/adapter_series_same_weight_FFN_Single_Inference_{}.pth'.format(iter_num))
        print('model save')

notebook_launcher(train, num_processes=1)