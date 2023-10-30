import numpy as np
from plotly.subplots import make_subplots 
import plotly.graph_objects as go


from sklearn.manifold import TSNE
from config import get_config

import json
import torch
import wandb
import pandas as pd
import torch.nn.functional as F
import collections

from data import ZuCo_dataset
from transformers import BartConfig, BartTokenizer, BertTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from model_decoding import BrainTranslator,BrainTranslatorNaive
import random

def build_sentence_tsne_data(data, use_features_ids=None, 
                             feature_name='input_embeddings', 
                             normalize="al", 
                             n_components=2, 
                             perplexity=30):
    vects = []
    for i in range(len(data)):
        if feature_name == 'input_embeddings':
            vects.append(np.mean((data[i][feature_name]).cpu().detach().numpy(), axis=0))
        elif feature_name is None:
        d = torch.squeeze(data[i])
            vects.append(np.mean(d.numpy(), axis=0))
        else:
            vects.append((data[i][feature_name]).cpu().detach().numpy())
    
    print("BEfore TSNE", np.array(vects).shape)
    if use_features_ids is None:
        use_features_ids = [i for i in range(len(vects[0]))]
    #vects = (np.array(vects) - np.mean(vects, axis=0)) / np.std(vects, axis=0)

    vects = np.array(vects)
    vects_sne = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(vects[:, use_features_ids])
    

    vects_sne = (np.array(vects_sne) - np.mean(vects_sne, axis=0)) / np.std(vects_sne, axis=0)

    print(vects_sne.shape)
    return vects_sne




def show_tsne_plot(vects_sne, data, subjects, d, output_path, config=None, show=False):
    df = pd.DataFrame({
               "subject": [data[i]["subject"] for i in range(len(data))],
               "content": [data[i]["content"][0] for i in range(len(data))],
               "task": [data[i]["task"] for i in range(len(data))],
    })

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Per Subject", "Per Sentence"))
    
    if config.tsne.plot_all_subjects:
        for s in subjects:
            indexes = list(df.index[df["subject"] == s])
            fig.add_trace(go.Scatter(x=vects_sne[indexes, 0],
                                    y=vects_sne[indexes, 1],
                                    mode='markers',    
                                    name=s,         
                                    legendgroup = '1',
                                    marker_color=subjects.index(s),
                                    text=[[data[i]["subject"], data[i]["content"]] for i in indexes]),
                row=1, col=1) # hover text goes here
        fig.add_trace(go.Scatter(x=vects_sne[:, 0],
                            y=vects_sne[:, 1],
                            mode='markers',             
                            marker_color=[d.index(data[i]["content"][0]) for i in range(len(data))],
                            text=[[data[i]["subject"][1], data[i]["content"][0]] for i in range(len(data))]),
                row=1, col=2)
        fig.update_layout(height=400, width=800,
                      xaxis1_range=[-2.5, 2.5],
                      yaxis1_range=[-2.5, 2.5],
                      xaxis2_range=[-2.5, 2.5],
                      yaxis2_range=[-2.5, 2.5],
                    title_text=f"{config.eeg_data.task_name[0]} TSNE Plot Perplexity {config.perplexity}")
    else:
        if config.tsne.plot_task_wise:
            if config.eeg_data.specific_subjects is None:

               config.eeg_data.specific_subjects = [random.choice(subjects)]
            print(f"\033[1m\033[93m Selected Subject : {config.eeg_data.specific_subjects} \033[0m")
            for s in config.eeg_data.specific_subjects:
                for i, task in enumerate(config.eeg_data.task_name):
                    indexes = list(df[(df["subject"] == s) & (df["task"] == task)].index)
                    fig.add_trace(go.Scatter(x=vects_sne[indexes, 0],
                                    y=vects_sne[indexes, 1],
                                    mode='markers',    
                                    name=task,         
                                    legendgroup = '1',
                                    marker_color=i,
                                    text=[[data[j]["subject"], data[j]["content"]] for j in indexes]),
                                 row=1, col=1) # hover text goes here
                    fig.add_trace(go.Scatter(x=vects_sne[indexes, 0],
                            y=vects_sne[indexes, 1],
                            mode='markers',             
                            marker_color=[d.index(data[i]["content"][0]) for i in indexes],
                            text=[[data[i]["subject"][1], data[i]["content"][0]] for i in indexes]),
                row=1, col=2)

                output_path = output_path.replace(".png", f"_{s}.png")
                fig.update_layout(height=400, width=800,
                      xaxis1_range=[-2.5, 2.5],
                      yaxis1_range=[-2.5, 2.5],
                      xaxis2_range=[-2.5, 2.5],
                      yaxis2_range=[-2.5, 2.5],
                    title_text=f"{s} TSNE Plot Perplexity {config.perplexity}")
                #if dist.get_rank() == 0:
                fig.write_image(output_path)

    fig.write_image(output_path)
    if show:
        fig.show()


def main():
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
    

    wandb.init(project="EEG-To-Text", group="Plot", name=f'{len(bands_choice)}_{eeg_type_choice}_{subject_choice}')
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
    subjects = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'])
               
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB'])
               
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB'])
             
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
        subjects.extend(['YFR', 'YDR', 'YAG', 'YAC', 'YDG', 'YAK', 'YFS', 'YHS'])
               

    
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', 
                            tokenizer, 
                            subject = subject_choice, 
                            eeg_type = eeg_type_choice, 
                            bands = bands_choice, 
                            setting = dataset_setting,
                            task_name = task_name)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}
    
    # List of sentences
    sentences = list(collections.Counter([test_set.inputs[i]["content"][0] for i in range(len(dev_dataset))]))


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
    model.eval()
    
    vectors = []
    
    with torch.no_grad():
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders['test']:
            
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            
            """replace padding ids in target_ids with -100"""
            #target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
            
            # forward
            encoded_embs = model.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_mask_invert_batch, )
            encoded_embs = F.relu(model.fc1(encoded_embs))
            vectors.append(encoded_embs)
            
        tsne_vects = build_sentence_tsne_data(vectors, feature_name=None)
        subj = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 'YFR', 'YDR', 'YAG', 'YAC', 'YDG', 'YAK', 'YFS', 'YHS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
        
        
    
    show_tsne_plot(tsne_vects, 
                   test_set.inputs, 
                   subjects,
                   sentences, 
                   out_path, 
                   config=config,  
                   show=False)





if __name__ == "__main__":
    main()