import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import wandb
from transformers import BertTokenizer, BertConfig,  BartTokenizer, BartForConditionalGeneration, BartConfig
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config
from sacrebleu.metrics import BLEU


def filtered_trained_layers(model, ckpt):

    b = {}
    for k in ckpt.keys():
        if "additional_encoder" in k:
            new_k = k[8:]
            b[new_k] = ckpt[k]
    b["fc1.weight"] = ckpt["encoder.fc.weight"]
    b["fc1.bias"] = ckpt["encoder.fc.bias"]

    model.load_state_dict(b, strict=False)



def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = './results/temp.txt', logger=None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path,'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders['test']:
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_string)
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100

            # forward
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

            """calculate loss"""
            # logits = seq2seqLMoutput.logits # 8*48*50265
            # logits = logits.permute(0,2,1) # 8*50265*48

            # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
            # NOTE: my criterion not used
            loss = seq2seqLMoutput.loss # use the BART language modeling loss


            # get predicted tokens
            # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size())
            logits = seq2seqLMoutput.logits # 8*48*50265
            # logits = logits.permute(0,2,1)
            # print('permuted logits size:', logits.size())
            probs = logits[0].softmax(dim = 1)
            # print('probs size:', probs.size())
            values, predictions = probs.topk(1)
            # print('predictions before squeeze:',predictions.size())
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','').replace('</s>','')
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # print('################################################')
            # print()

            sample_count += 1
            # statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
            # print('[DEBUG]loss:',loss.item())
            # print('#################################')


    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    print(target_tokens_list[0], type(target_tokens_list))
    print()
    print(pred_tokens_list[0])   

    tgt_decoded_list = [[tgt.split()] for tgt in target_string_list ]
    pred_decoded_list = [p.split() for p in pred_string_list]

    table = wandb.Table(columns=["Metric", "Value"])
    sb_table = wandb.Table(columns=["Metric", "Value"])
    tok_table = wandb.Table(columns=["Metric", "Value"])
    """igalculate corpus bleu score """
    bleu = BLEU()
    sc = bleu.corpus_score(pred_string_list, [[t] for t in target_string_list])
    print("SacreBLEU: ", sc)
    sb_table.add_data(f"SacreBLEU-1 ", f"{sc.precisions[0]}")
    sb_table.add_data(f"SacreBLEU-2 ", f"{sc.precisions[1]}")
    sb_table.add_data(f"SacreBLEU-3 ", f"{sc.precisions[2]}")
    sb_table.add_data(f"SacreBLEU-4 ", f"{sc.precisions[3]}")
    sb_table.add_data(f"SacreBLEU ", f"{sc.score}")


    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_raw_score = corpus_bleu(tgt_decoded_list, pred_decoded_list, weights = weight)
        tok_table.add_data(f'BLEU-{len(list(weight))} tok', f'{corpus_bleu_score}')
        table.add_data(f'BLEU-{len(list(weight))} raw', f'{corpus_raw_score}')
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print(f'corpus Raw BLEU-{len(list(weight))} score:', corpus_raw_score)
    wandb.log({"RawBLEU Scores": table})
    wandb.log({"tokBLEU Scores": tok_table})
    wandb.log({"SacreBLEU Scores": sb_table})
    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list,target_string_list, avg = True)
    rg_table = wandb.Table(columns=["Metric", "f1 score", "precision", "recall"])
    for rg in rouge_scores.keys():
        rg_table.add_data(rg, f'{rouge_scores[rg]["f"]*100:.4f}', f'{rouge_scores[rg]["p"]*100:.4f}', f'{rouge_scores[rg]["r"]*100:.4f}')
    print(rouge_scores)
    wandb.log({"ROUGE Scores": rg_table})
    wandb.finish()

if __name__ == '__main__': 
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))
    
    batch_size = 1
    
    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    if "7_feats" in args['config_path']:
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1']
    else:
        bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    

    wandb.init(project="EEG-To-Text", group="Eval", name=f'{len(bands_choice)}_{eeg_type_choice}_{subject_choice}')
    model_name = training_config['model_name']
    # model_name = 'BrainTranslator'
    # model_name = 'BrainTranslatorNaive'
    from datetime import datetime

    now = datetime.now() 
    
    seed_val = 312
    output_all_results_path = f'./results/{now.strftime("%Y-%m-%d")}_{task_name}-{model_name}_{seed_val}_decoding_results.txt'
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')


    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=len(bands_choice), additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BertGeneration':
        from transformers import BertLMHeadModel
        pretrained = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 768, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    if args["eval_trained_diffusion"]:
        print('[INFO] We ARE Now evaluating the pretrained diffusion model: ', args['diffusion_checkpoint_path'])
        diffusion_checkpoint_path = args['diffusion_checkpoint_path']
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        ckpt = torch.load(diffusion_checkpoint_path, map_location=dev)['model_dict']
        filtered_trained_layers(model, ckpt)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path)
