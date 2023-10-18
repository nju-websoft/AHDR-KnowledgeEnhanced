# coding=utf-8
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from tqdm import trange
import os
import random
import torch
from utils import save_dataset, read_dataset, set_seed, save_model
import json
import argparse
from torch import nn
from collections import OrderedDict
from tqdm import tqdm
import ast
from monobert_eval import simple_eval

device = torch.device("cuda:0")

class MonoT5(nn.Module):
    def __init__(self, model_path):
        super(MonoT5, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        dim = self.model.config.hidden_size
        self.scoring = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.token_false_id = tokenizer.get_vocab()[token_false]
        self.token_true_id = tokenizer.get_vocab()[token_true]
        self.lm_head = self.model.lm_head


    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            t5_output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                      labels=labels)
            loss = t5_output.loss
            return loss
        else:
            t5_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                attention_mask=attention_mask,
                do_sample=False,
                # output_hidden_states=True,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            decoder_hidden_states = t5_output['decoder_hidden_states']
            decoder_hidden_states = decoder_hidden_states[0][-1]
            logits = self.lm_head(decoder_hidden_states)
            logits = logits[:, -1, :]

            logits = logits[:, self.token_true_id]
            return logits

def clean_text(text):
    text = text.replace('[CLS] ', '')
    text = text.replace('[SEP] ', '')
    return text

def tokenizer_fun(input_ids, max_len):
    encoding = tokenizer(input_ids,
                         padding='longest',
                         max_length=max_len,
                         truncation='longest_first',
                         return_tensors="pt")
    ids = encoding.input_ids.to(device)
    mask = encoding.attention_mask.to(device)
    return ids, mask

def get_input_feature_train(features, max_source_length, max_pos, max_neg):
    input_texts, labels = [], []
    for sample in features:
        positive_ctxs = sample['positive_ctxs']
        hard_negative_ctxs = sample['hard_negative_ctxs']
        if len(positive_ctxs) > max_pos:
            positive_ctxs = positive_ctxs[:max_pos]
        if len(hard_negative_ctxs) > max_neg:
            hard_negative_ctxs = hard_negative_ctxs[:max_neg]
        question = sample['question']
        for positive_ctx in positive_ctxs:
            text = positive_ctx["text"]
            input_texts.append(f'Question: {question} Context: {clean_text(text)}')
            labels.append('true')
        for hard_negative_ctx in hard_negative_ctxs:
            text = hard_negative_ctx["text"]
            input_texts.append(f'Question: {question} Context: {clean_text(text)}')
            labels.append('false')

    input_ids, input_masks = tokenizer_fun(input_texts, max_source_length)
    target_ids, _ = tokenizer_fun(labels, max_source_length)
    target_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
        target_ids
    ]
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)
    return input_ids, input_masks, target_ids


def get_input_feature_inference(sample, max_source_length):
    input_texts = []
    ctxs = sample['ctxs']
    question = sample['question']
    c_ids = []
    for ctx in ctxs:
        c_ids.append(ctx['c_id'])
        text = ctx["text"]
        input_texts.append(f'Question: {question} Context: {clean_text(text)}')
    input_ids, input_masks = tokenizer_fun(input_texts, max_source_length)
    return input_ids, input_masks, c_ids


def get_input_feature_evaluate(sample, max_source_length):
    input_texts, labels = [], []
    positive_ctxs = sample['positive_ctxs']
    hard_negative_ctxs = sample['hard_negative_ctxs']
    question = sample['question']
    q_rels = {}
    c_id = 0
    for positive_ctx in positive_ctxs:
        q_rels[str(c_id)] = 1
        c_id += 1
        text = positive_ctx["text"]
        input_texts.append(f'Question: {question} Context: {clean_text(text)}')
        labels.append('true')
    for hard_negative_ctx in hard_negative_ctxs:
        q_rels[str(c_id)] = 0
        c_id += 1
        # input_texts.append([
        #     question,
        #     hard_negative_ctx['text']
        # ])
        text = hard_negative_ctx["text"]
        input_texts.append(f'Question: {question} Context: {clean_text(text)}')
        labels.append('false')

    input_ids, input_masks = tokenizer_fun(input_texts, max_source_length)
    target_ids, _ = tokenizer_fun(labels, max_source_length)
    target_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
        target_ids
    ]
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)

    return input_ids, input_masks, target_ids, q_rels



@torch.no_grad()
def inference(model, test_examples, q_rels, eval_batch_size, max_len):
    model.eval()
    res_dict = {}
    for sample in tqdm(test_examples):
        q_id = sample['q_id']
        input_ids, attention_mask, c_ids = get_input_feature_inference(sample, max_len)
        pred_scores = []
        step_count = len(input_ids) // eval_batch_size
        if step_count * eval_batch_size < len(input_ids):
            step_count += 1
        for step in range(step_count):
            beg_index = step * eval_batch_size
            end_index = min((step + 1) * eval_batch_size, len(input_ids))
            input_ids_bt = input_ids[beg_index:end_index]
            attention_mask_bt = attention_mask[beg_index:end_index]

            input_ids_bt = torch.tensor(input_ids_bt, dtype=torch.long).to(device)
            attention_mask_bt = torch.tensor(attention_mask_bt, dtype=torch.long).to(device)

            logits = model(input_ids_bt, attention_mask_bt)
            logits = logits.tolist()
            pred_scores += logits
        pred_scores_id = [[c_id, pred_score] for c_id, pred_score in zip(c_ids, pred_scores)]
        pred_scores_id = sorted(pred_scores_id, key=lambda x: x[1], reverse=True)
        res_dict[q_id] = pred_scores_id
    q_rels_cut = {}
    for q_id in res_dict.keys():
        q_rels_cut[q_id] = q_rels[q_id]

    result_score = simple_eval(q_rels_cut, res_dict, k=10)
    return result_score, res_dict

@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, max_len):
    model.eval()
    q_rels = {}
    res_dict = {}
    q_id = 0
    for sample in tqdm(test_examples):
        input_ids, attention_mask, labels, q_rels_item = get_input_feature_evaluate(sample, max_len)
        q_rels[str(q_id)] = q_rels_item
        pred_scores = []
        step_count = len(input_ids) // eval_batch_size
        if step_count * eval_batch_size < len(input_ids):
            step_count += 1
        for step in range(step_count):
            beg_index = step * eval_batch_size
            end_index = min((step + 1) * eval_batch_size, len(input_ids))
            input_ids_bt = input_ids[beg_index:end_index]
            attention_mask_bt = attention_mask[beg_index:end_index]

            input_ids_bt = torch.tensor(input_ids_bt, dtype=torch.long).to(device)
            attention_mask_bt = torch.tensor(attention_mask_bt, dtype=torch.long).to(device)

            logits = model(input_ids_bt, attention_mask_bt)
            logits = logits.tolist()
            pred_scores += logits

        pred_scores_id = [[c_id, pred_score] for c_id, pred_score in enumerate(pred_scores)]
        pred_scores_id = sorted(pred_scores_id, key=lambda x: x[1], reverse=True)
        res_dict[str(q_id)] = pred_scores_id
        q_id += 1
    result_score = simple_eval(q_rels, res_dict, k=10)
    return result_score, res_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",
                        default='0',
                        type=str)
    parser.add_argument("--task",
                        default='ACORDAR',
                        type=str)
    parser.add_argument("--model_name",
                        default='castorini/monot5-large-msmarco',
                        type=str)
    parser.add_argument("--save_model",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--train_paths",
                        default=['./data/train/acordar1_metadata_split_0.json',
                                 './data/train/acordar1_metadata_split_1.json',
                                 './data/train/acordar1_metadata_split_2.json',
                                 './data/train/acordar1_metadata_split_4.json'],
                        type=list)
    parser.add_argument("--dev_paths",
                        default=['./data/train/acordar1_metadata_split_3.json'],
                        type=list)
    parser.add_argument("--test_path",
                        default='./data/test/ntcir_BM25_top10.json',
                        type=str)
    parser.add_argument("--test_qrels_path",
                        default='./data/qrels/ntcir_merge_qrels.json',
                        type=str)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--max_train_pos",
                        default=3,
                        type=int)
    parser.add_argument("--max_train_neg",
                        default=3,
                        type=int)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    token_true = '▁true'
    token_false = '▁false'
    result_score_all = {}
    args.test_qrels_path = './data/qrels/acordar1_test_qrels.json'
    test_qrels = read_dataset(args.test_qrels_path)
    for fold_i in range(5):
        t1 = fold_i
        t2 = (fold_i + 1) % 5
        t3 = (fold_i + 2) % 5
        t4 = (fold_i + 3) % 5
        t5 = (fold_i + 4) % 5
        args.train_paths = [f'./data/train/acordar1_metadata_split_{t1}.json',
                            f'./data/train/acordar1_metadata_split_{t2}.json',
                            f'./data/train/acordar1_metadata_split_{t3}.json', ]
        args.dev_paths = [f'./data/train/acordar1_metadata_split_{t4}.json']
        args.test_path = f'./data/test/acordar1_BM25_top10_split_{t5}.json'


        only_eval = args.only_eval
        debug = args.debug
        test_name = args.test_path.split('/')[-1].replace('.json', '')
        encoder_name = args.model_name.split('/')[-1].split('-')[0]
        config_name = f'monoT5_{args.task}_{encoder_name}'
        parameter_name = f'fold_{t5}_lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}_{args.max_train_pos}_{args.max_train_neg}' \
                         f'_ga_{args.gradient_accumulation_steps}'
        output_model_path = f'./outputs/{config_name}/{parameter_name}/'
        path_save_result = f'./results/{config_name}/{parameter_name}/'

        os.makedirs(path_save_result, exist_ok=True)
        set_seed(args.seed)

        train_examples, dev_examples = [], []

        for train_path in args.train_paths:
            train_examples += read_dataset(train_path)
        for dev_path in args.dev_paths:
            dev_examples += read_dataset(dev_path)

        test_examples = read_dataset(args.test_path)

        if debug:
            train_examples = train_examples[:20]
            dev_examples = dev_examples[:20]
            test_examples = test_examples[:20]

        train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = MonoT5(args.model_name).to(device)

        print('# parameters:', sum(param.numel() for param in model.parameters()))
        print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                          "bs": args.train_batch_size,
                          'gradient_accumulation_steps': args.gradient_accumulation_steps,
                          "epoch": args.epoch_num,
                          "train_size": len(train_examples),
                          "dev_size": len(dev_examples),
                          "test_size": len(test_examples),
                          'max_len': args.max_len,
                          'output_model_path': output_model_path,
                          'path_save_result': path_save_result,
                          'init_checkpoint': args.init_checkpoint,
                          'max_train_pos': args.max_train_pos,
                          'max_train_neg': args.max_train_neg
                          }, indent=2))
        if args.init_checkpoint:
            init_checkpoint = f'{output_model_path}/pytorch_model.bin'
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            model_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            for k in list(model_dict.keys()):
                name = k
                if k.startswith('module.bert.bert.'):
                    name = k.replace("module.bert.", "")
                new_state_dict[name] = model_dict[k]
                del model_dict[k]
            model.load_state_dict(new_state_dict, False)
            print('init from:', init_checkpoint)

        if only_eval:
            result_score_test, results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
                                                        args.max_len)
            test_scores = result_score_test['mean']
            for key in test_scores.keys():
                if key not in result_score_all:
                    result_score_all[key] = test_scores[key]
                else:
                    result_score_all[key] = result_score_all[key] + test_scores[key]

            print('test scores:', json.dumps(result_score_test['mean'], indent=2))
            save_dataset(path_save_result + '/test.json', results_test)

        else:
            result_score_test, results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
                                                        args.max_len)
            print("init score:", result_score_test['mean'])
            warm_up_ratio = 0.05
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            t_total = args.epoch_num * (len(train_examples) // train_batch_size // args.gradient_accumulation_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=int(warm_up_ratio * (t_total)),
                                                        num_training_steps=t_total)
            step_count, step_all, early_stop = 0, 0, 0

            best_test_ndcg = 0
            best_dev_ndcg = 0
            best_dev_result, best_test_result = None, None
            for epoch in range(args.epoch_num):
                tr_loss, nb_tr_steps = 0, 0.1
                early_stop += 1
                order = list(range(len(train_examples)))
                random.seed(args.seed + epoch)
                random.shuffle(order)
                model.train()
                step_count = len(train_examples) // train_batch_size
                if step_count * train_batch_size < len(train_examples):
                    step_count += 1
                step_trange = trange(step_count)
                for step in step_trange:
                    step_all += 1
                    beg_index = step * train_batch_size
                    end_index = min((step + 1) * train_batch_size, len(train_examples))
                    order_index = order[beg_index:end_index]
                    batch_example = [train_examples[index] for index in order_index]
                    input_ids, attention_mask, labels = get_input_feature_train(
                        batch_example, args.max_len, max_pos=args.max_train_pos, max_neg=args.max_train_neg)
                    loss = model(input_ids, attention_mask, labels)
                    loss = loss.mean()
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                        round(tr_loss / nb_tr_steps, 4)) + f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
                    step_trange.set_postfix_str(loss_show)
                result_score_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, args.max_len)
                ndcg_cut_5_mean = result_score_dev['mean']['ndcg_cut_5_mean']
                print('dev scores:', json.dumps(result_score_dev['mean'], indent=2))
                save_dataset(path_save_result + '/dev.json', results_dev)
                if ndcg_cut_5_mean > best_dev_ndcg:
                    print('new best')
                    best_dev_result = result_score_dev
                    best_dev_ndcg = ndcg_cut_5_mean
                    save_model(output_model_path, model, optimizer)
                    result_score_test, results_test = inference(model, test_examples, test_qrels, args.eval_batch_size,
                                                                args.max_len)
                    test_scores = result_score_test['mean']
                    for key in test_scores.keys():
                        if key not in result_score_all:
                            result_score_all[key] = test_scores[key]
                        else:
                            result_score_all[key] = result_score_all[key] + test_scores[key]

                    print('test scores:', json.dumps(result_score_test['mean'], indent=2))
                    save_dataset(path_save_result + '/test.json', results_test)


    for key in result_score_all.keys():
        result_score_all[key] = round(result_score_all[key] / 5, 4)
    print('fold avg:', json.dumps(result_score_all, indent=2))
    res_dict_all = {}

    for fold_i in range(5):
        t5 = (fold_i + 4) % 5
        test_name = args.test_path.split('/')[-1].replace('.json', '')
        encoder_name = args.model_name.split('/')[-1].split('-')[0]
        config_name = f'monoT5_{args.task}_{encoder_name}'
        parameter_name = f'fold_{t5}_lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}_{args.max_train_pos}_{args.max_train_neg}' \
                         f'_ga_{args.gradient_accumulation_steps}'
        output_model_path = f'./outputs/{config_name}/{parameter_name}/'
        path_save_result = f'./results/{config_name}/{parameter_name}/'
        res_dict = read_dataset(path_save_result + '/test.json')
        res_dict_all.update(res_dict)

    result_score = simple_eval(test_qrels, res_dict_all, k=10)
    ndcg_cut_5_mean = result_score['mean']['ndcg_cut_5_mean']
    ndcg_cut_10_mean = result_score['mean']['ndcg_cut_10_mean']
    ndcg_cut_5_mean = ndcg_cut_5_mean * (483 / 493)
    ndcg_cut_10_mean = ndcg_cut_10_mean * (483 / 493)
    print('ndcg_cut_5_mean:', round(ndcg_cut_5_mean, 4))
    print('ndcg_cut_10_mean:', round(ndcg_cut_10_mean, 4))
