import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    if case == 'train_decoding': 
        # args config for training EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')
        parser.add_argument('-nwork', '--num_workers', type=int, default=8) 
        parser.add_argument('-m', '--model_name', help='choose from {BrainTranslator, BrainTranslatorNaive}', default = "BrainTranslator" ,required=True)
        parser.add_argument('-t', '--task_name', help='choose from {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2}', default = "task1", required=True)
        
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')

        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false')

        parser.add_argument('-ne1', '--num_epoch_step1', type = int, help='num_epoch_step1', default = 20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type = int, help='num_epoch_step2', default = 30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type = float, help='learning_rate_step1', default = 0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type = float, help='learning_rate_step2', default = 0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/decoding', required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')

        parser.add_argument('--wb_logger', help="activate wb logger", action='store_true')
        parser.add_argument('--data_folder', help="", default='/disk/scratch/s2616972')
        parser.add_argument('--tok', help="tokenizer used", default="facebook/bart-large")

        args = parser.parse_args()

    elif case == 'train_sentiment_baseline':
        # args config for training EEG-based sentiment baselines
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')
        
        parser.add_argument('-m', '--model_name', help='choose from {BaselineMLP, BaselineLSTM, NaiveFinetuneBert}', default = "NaiveFinetuneBert" ,required=True)
        parser.add_argument('-ne', '--num_epoch', type = int, help='num_epoch', default = 30, required=True)
        parser.add_argument('-lr', '--learning_rate', type = float, help='learning_rate', default = 0.00001, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/eeg_sentiment', required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        args = vars(parser.parse_args())
        
    elif case == 'train_sentiment_textbased': 
        # args config for training text-based sentiment classification models
        parser = argparse.ArgumentParser(description='Specify config args for training text-based sentiment classifiers')
        parser.add_argument('-d', '--dataset_name', help='zero-shot setting: using external dataset from stanford sentiment treebank, pass in SST; to use ZuCo\'s own text-sentiment pairs, pass in ZuCo', default = "SST" ,required=True)
        parser.add_argument('-m', '--model_name', help='choose from {pretrain_Bert, pretrain_RoBerta, pretrain_Bart}', default = "pretrain_Bart" ,required=True)
        parser.add_argument('-ne', '--num_epoch', type = int, help='num_epoch', default = 20, required=True)
        parser.add_argument('-lr', '--learning_rate', type = float, help='learning_rate', default = 0.0001, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/text_sentiment_classifier', required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        args = vars(parser.parse_args())
        
    elif case == 'eval_decoding':
        # args config for evaluating EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-To-Text decoder')
        parser.add_argument("--eval_trained_diffusion", action='store_true')
        parser.add_argument("--diffusion_checkpoint_path")
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint' ,required=True)
        parser.add_argument('-conf', '--config_path', help='specify training config json' ,required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        args = vars(parser.parse_args())
        
    elif case == 'eval_sentiment':
        # args config for sentiment classification models
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-based sentiment classification, including Zero-shot pipeline')
        # choose model_name = 'ZeroShotSentimentDiscovery' to evaluate Zero-shot pipeline
        parser.add_argument('-m', '--model_name', help='choose from {BaselineMLP, BaselineLSTM, NaiveFinetuneBert, FinetunedBertOnText, FinetunedRoBertaOnText, FinetunedBartOnText, ZeroShotSentimentDiscovery}', default = "ZeroShotSentimentDiscovery" ,required=True)
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint' ,required=False) # required if NOT evaluating Zero-shot pipeline
        parser.add_argument('-conf', '--config_path', help='specify model config json' ,required=False) # required if NOT evaluating Zero-shot pipeline
        parser.add_argument('-checkpoint_DEC', '--decoder_checkpoint_path', help='specify decoder checkpoint for Zero-shot pipeline ', required=False) # required if evaluating Zero-shot pipeline
        parser.add_argument('-checkpoint_CLS', '--classifier_checkpoint_path', help='specify classifier checkpoint for Zero-shot pipeline ', required=False) # required if evaluating Zero-shot pipeline
        parser.add_argument('-conf_DEC', '--decoder_config_path', help='specify decoder config json' ,required=False) # required if evaluating Zero-shot pipeline
        parser.add_argument('-conf_CLS', '--classifier_config_path', help='specify classifier config json' ,required=False) # required if evaluating Zero-shot pipeline
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        args = vars(parser.parse_args())

    return args

from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
class DecoderConfig:

    model_name = BrainTranslator
    task_name = "task1_task2_taskNRv2"

    one_step = True
    two_step = False
    
    if one_step:
        skip_step_one=True
    else:
        skip_step_one = False

    pretrained = False
    rand_init = True

    if rand_init:
        use_random_init=True
    else:
        use_random_init = False

    load_step1_checkpoint = True
    not_load_step1_checkpoint = False

    num_epoch_step1 = 20
    num_epoch_step2 = 30
    learning_rate_step1 = 0.00005
    learning_rate_step2 = 0.0000005

    batch_size = 32
    save_path = './checkpoints/decoding'
    subjects = 'ALL'

    eeg_type = 'GD'
    eeg_bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    cuda = 'cuda:0'
    wb_logger = True
    data_folder = '/disk/scratch/s2616972'
    num_workers = 8

def self_update(cfg):
    if cfg.one_step:
        cfg.skip_step_one = True
    else:
        cfg.skip_step_one = False

    if cfg.rand_init:
        cfg.use_random_init=True
    else:
        cfg.use_random_init = False

def update_config(args, config):
    for attr in config.__dict__:
        if not attr.startswith('__'):
            if hasattr(args, attr):
                if getattr(args, attr) != None:
                    setattr(config, attr, getattr(args, attr))
    self_update(config)
    return config

if __name__ == '__main__':
    args = get_config('train_decoding')


    cfg = DecoderConfig
    cfg = update_config(args, cfg)

    #logger = wandb_logger(cfg) if cfg.wb_logger else None
    import json
    args = vars(cfg)

    ''' config param'''
    dataset_setting = 'unique_sent'
    
    num_epoch_step1 = args['num_epoch_step1']
    num_epoch_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    
    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # with no additional transformers
    # model_name = 'BrainTranslator' 
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']

    save_path = args['save_path']

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4
        
    print(f'[INFO]using model: {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epoch_step1}_{num_epoch_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epoch_step1}_{num_epoch_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    
    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt' 


    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    with open(output_checkpoint_name_best.replace(".pt",".json").replace( "checkpoints", "config"), "w")as out_config:
        config_json = dict(args)
        sjson = dict()
        ks = config_json.keys()
        for k in ks:
            if not k.startswith('__'):
                sjson[k] = config_json[k]
        json.dump(sjson, out_config, indent = 4)

