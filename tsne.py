import os
import hydra
import torch
import logging
import random
import pandas as pd

import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from utils.logger import WandbCustLogger
import torch.nn.functional as F

from utils.utils import load_states_from_checkpoint
from dataset.eeg_s2s_dataset import load_zuco_data, load_preprocessed_zuco_data, load_zuco_test_data, S2S_dataset
from dataset.s2s_dataset import load_jsonl_data
from dataset.tokenizer_utils import create_tokenizer
from model.diffusion.create_model import create_model, create_gaussian_diffusion
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import corpus_bleu
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import collections



logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

subj = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 'YFR', 'YDR', 'YAG', 'YAC', 'YDG', 'YAK', 'YFS', 'YHS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']


def build_sentence_tsne_data(data, use_features_ids=None, feature_name='input_embeddings', normalize="al", n_components=2, perplexity=30):
  
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
               import random
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
    if dist.get_rank() == 0:
      fig.write_image(output_path)
    if show:
        fig.show()

def denoised_fn_round(config, emb_model, text_emb, t):
    down_proj_emb = emb_model.weight  # (vocab_size, embed_dim)

    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # (vocab, 1)
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # (emb_dim, bs*seqlen)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # (bs*seqlen, 1)
            # down_proj_emb: (vocab, emb_dim), text_emb_t:(emb_dim, bs*seqlen)
            # a+b automatically broadcasts to the same dimension i.e. (vocab, bs*seqlen)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb, text_emb_t) 
            dist = torch.clamp(dist, 0.0, np.inf)  # Limit the value of input to [min, max].
        # Select the smallest distance in the vocab dimension, 
        # that is, select bs*seq_len most likely words from all vocabs.
        topk_out = torch.topk(-dist, k=1, dim=0)

        return topk_out.values, topk_out.indices  # logits, token_id (1, bs*seq_len)

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]  # (bs*seq_len,)
    new_embeds = emb_model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds


def split_data(data, log=False):
    shard_size = len(data) // dist.get_world_size()
    start_idx = dist.get_rank() * shard_size
    end_idx = start_idx + shard_size
    if dist.get_rank() == dist.get_world_size() - 1:
        end_idx = len(data)
    data_piece = data[start_idx: end_idx]
    
    if log:
        logger.info(f'generation for {len(data_piece)} text from idx {start_idx} to {end_idx}')
    
    return data_piece


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    config.exp.dir = os.path.join(config.exp.root, config.data.name, config.exp.name)
    generate_path = os.path.join(config.exp.dir, str(config.load_step))
    if config.load_from_ema:
        generate_path += ('_ema_' + str(config.ema_rate))
    if config.clip_denoised:
        generate_path += '_clip_denoised_'
    if config.infer_self_condition:
        generate_path += '_selfcond_'
    if config.skip_sample:
        generate_path += '_skip_'
    if config.ddim_sample:
        generate_path += '_ddim_'

    if config.schedule_sampler == 'xy_uniform':
        generate_path += ('_xy_' + str(config.gen_timesteps))
    else:
        generate_path += ('_un_' + str(config.skip_timestep))

    if (local_rank == 0) and (not os.path.exists(generate_path)):
        os.makedirs(generate_path)
   
    config.gen_timesteps = int(config.gen_timesteps)
    torch.cuda.set_device(local_rank)  # ddp setting
    dist.init_process_group(backend="nccl")  # ddp setting
    
    config.logger.name = str(config.logger.name)
    config.logger.name += f"_gen_{config.exp.name}_mpl{config.max_pos_len}"
    from datetime import datetime
    now = datetime.now()
    config.logger.name = now.strftime("%Y-%m-%d_") + config.logger.name
    config.group = "Generate_baseline"
    wb_logger = WandbCustLogger(config)
    wb_logger.init_table("Tokens BLEU", columns=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
    
    set_seed(config.exp.seed + int(dist.get_rank()))  # seed setting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load tokenizer
    if config.data.name in ['iwslt14', 'iwslt14_tok']:
        tokenizer = None
        if config.use_bpe:
            tokenizer = create_tokenizer(path=f'./data/{config.data.name}/')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    if tokenizer == None:
        vocab_size = config.vocab_size
    else:
        vocab_size = tokenizer.vocab_size
        if config.data.name in ['iwslt14', 'iwslt14_tok']:
            if config.use_bpe:
                config.pad_value = tokenizer.get_vocab()['<pad>']
            # else use by fairseq
        else:
            config.pad_value = tokenizer.pad_token_id

    # define model and diffusion
    model, diffusion = create_model(config, vocab_size), create_gaussian_diffusion(config)
    model.to(device).eval()

    # load trained model
    if config.load_from_ema:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'ema_{config.ema_rate}_checkpoint-{config.load_step}')
    else:
        eval_model_path = os.path.join(
            config.exp.dir, 'model', f'model_checkpoint-{config.load_step}')
    model_saved_state = load_states_from_checkpoint(eval_model_path, dist.get_rank())
    if not config.load_eeg_checkpoint:
        model.load_state_dict(model_saved_state.model_dict)
    else:
        eval_model_path = os.path.join(config.exp.root, config.data.name)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if dist.get_rank() == 0:
        logger.info(f'the parameter count is {pytorch_total_params}')
        
    if dist.get_world_size() > 1:
        model = DDP(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=False,
        )

    if config.ddim_sample:
        sample_fn = (diffusion.ddim_sample_loop)
    else:
        sample_fn = (diffusion.p_sample_loop)

    if dist.get_world_size() > 1:
        emb_model = model.module.word_embedding
    else:
        emb_model = model.word_embedding

    if config.model.mode == 's2s':
        if dist.get_rank() == 0:
            print(f"start generate query from dev dataset, for every passage, we generate {config.num_samples} querys...")
            logger.info("***** load " + config.data.name + " dev dataset*****")
            
        dev_dataset = load_zuco_test_data(config, tokenizer)        
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=config.batch_size, 
            drop_last=False, pin_memory=True, num_workers=config.num_workers, 
            collate_fn=S2S_dataset.get_collate_fn(config)
        )    
        sentences = list(collections.Counter([dev_dataset.data[i]["content"][0] for i in range(len(dev_dataset))]))

        for i in range(config.num_samples):
            torch.cuda.empty_cache()
            each_tgt_sample_list = []
            each_sample_list = []
            each_sample_token_list = []
            each_sample_pred_token_list = []
            if dist.get_rank() == 0:
                print(f"start sample {i+1} epoch...")
             
            encoded_eeg = []            

            for _, batch in enumerate(tqdm(dev_dataloader)):
                # with torch.no_grad():
                #print(f"\033[96m\033[1m SRC input {batch['src_input_ids'].shape} MASK INVERT {batch['src_attention_mask_invert'].shape}\033[0m")
                encoder_hidden_states = model.module.encoder.additional_encoder(
                    batch['src_input_ids'].float().cuda(), 
                    src_key_padding_mask=batch['src_attention_mask_invert'].float().cuda() #.unsqueeze(1),
                ) #.last_hidden_state  # [bs, seq_len, hz]
                encoder_hidden_states = F.relu(model.module.encoder.fc(encoder_hidden_states))
                if dist.get_rank() == 0:
                    logger.info(f"\033[1m\033]96m src input ids {batch['src_input_ids'].shape} encoder_hidden_states {encoder_hidden_states.shape}  \033[0m")
                with torch.no_grad():
                    encoded_eeg.append(encoder_hidden_states.cpu().detach())
               
                
            output_path = os.path.join(generate_path, 'num' + str(i))
            tgt_output_path = os.path.join(generate_path, 'num' + str(i))
            if dist.get_rank() == 0:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
            dist.barrier()
            
            if config.tsne.plot_task_wise:
                out_path = os.path.join(f"{config.eeg_data.task_name[0]}_"+ "rank" + str(dist.get_rank())+"_seed_" + str(config.exp.seed) + "perplexity" + str(config.perplexity) + "_tsne_plot.png")
            out_path = os.path.join(
                output_path, f"Additional_enc_ONLY_{config.eeg_data.task_name[0]}_"+ "rank" + str(dist.get_rank())+"_seed_" + str(config.exp.seed) + "perplexity" + str(config.perplexity) + "_tsne_plot.png")
            if config.load_eeg_checkpoint:
                print("\n\n CONFIG LOAD EEG CHECKPOINT IS TRUE")
                out_path = os.path.join(config.exp.root, config.data.name, f"{config.eeg_data.task_name[0]}_"+ "rank" + str(dist.get_rank())+"_seed_" + str(config.exp.seed) + "perplexity" + str(config.perplexity) + "_tsne_plot.png")
            with torch.no_grad():
                vects_tsne = build_sentence_tsne_data(encoded_eeg, feature_name=None, perplexity=config.perplexity)
                subjects = []
                if "task1-SR" in config.eeg_data.task_name:
                    subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'])
                if "task2-NR" in config.eeg_data.task_name:
                    subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB'])
                if "task3-TSR" in config.eeg_data.task_name:
                    subjects.extend(['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB'])
                if "task2-NR-2.0" in config.eeg_data.task_name:
                    subjects.extend(['YFR', 'YDR', 'YAG', 'YAC', 'YDG', 'YAK', 'YFS', 'YHS'])
                show_tsne_plot(vects_tsne, dev_dataset.data, subjects,sentences, out_path, config=config,  show=False)
            print(f"SAMPLE {i}; Output figure was saved as file {out_path}")
    else:
        return NotImplementedError


if __name__ == "__main__":
    main()
