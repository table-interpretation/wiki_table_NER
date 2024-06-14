import os
import openai
import pandas as pd
import csv
import json
import re
import torch
import time
from collections import defaultdict
from utils import llm_responce, tab_to_csv, plot_conf_matrix, get_label_dict, save_metrics, generate_example
from gpt_utils import read_gpt_config
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

seed_nr = 42
generator = torch.Generator().manual_seed(seed_nr)
print(torch.cuda.is_available())


def init_llama_model(model_name):
    """
    Initialization for the llama2 pipeline
    """
    if model_name == "llama-2-7b":
        # model_dir = "../llama/llama-2-7b"
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')
        device = torch.device('cuda')

        pipeline = transformers.pipeline("text-generation",
                                         model=model,
                                         tokenizer=tokenizer,
                                         torch_dtype=torch.float16,
                                         device_map='auto',
                                         )

        return pipeline


def get_llama_responce(pipeline, shot, prompt, tab_id):
    print("Start Llama")
    responce = pipeline(prompt,
                        temperature=0.2,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        max_length=4096)

    with open('output/Llama/log_file_{}_{}rand.txt'.format("llama7b", shot), 'a') as file:
        file.write("{}:{}".format(tab_id, responce))
        file.write("\n")

    return responce


def get_llm_responce(pipeline, model_name, shot, prompt, tab_id):
    if model_name == "llama-2-7b":
        return get_llama_responce(pipeline, shot, prompt, tab_id)
    else:
        return get_gpt_answer(model_name, shot, prompt, tab_id)


def get_gpt_answer(model_name, shot, prompt, tab_id):
    """
    Prompting GPT models and parsing the output.
    If failed to parse, save the output to log file.
    """
    read_gpt_config(model_name)
    response = {}
    try:
        if model_name == "gpt-35-turbo-instruct":
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=4000,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)

        elif model_name in ["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4"]:
            response = openai.ChatCompletion.create(
                engine=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an NER expert. Your task is to label entities in a table with given types."},
                    {"role": "user",
                     "content": "{}".format(prompt)}],
                temperature=0,
                max_tokens=2500,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)

        else:
            response = openai.ChatCompletion.create(
                engine=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are an NER expert. Your task is to label entities in a table with given types."},
                    {"role": "user",
                     "content": "{}".format(prompt)}],
                temperature=0,
                max_tokens=2500,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)

    except openai.error.Timeout as e:
        # Handle timeout error, e.g. retry or log
        print(f"OpenAI API request timed out: {e}")
        time.sleep(2)
        pass
    except openai.error.APIError as e:
        # Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        time.sleep(2)
        pass
    except openai.error.APIConnectionError as e:
        # Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        time.sleep(2)
        pass
    except openai.error.InvalidRequestError as e:
        # Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request, {tab_id} was invalid: {e}")
        time.sleep(2)
        pass
    except openai.error.AuthenticationError as e:
        # Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        time.sleep(2)
        pass
    except openai.error.PermissionError as e:
        # Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        time.sleep(2)
        pass
    except openai.error.RateLimitError as e:
        # Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(5)
        pass

    if len(response) > 0:
        extraction_dict = {}
        choices = response["choices"][0]

        try:
            if model_name == "gpt-35-turbo-instruct":
                text = choices["text"]
            else:
                text = choices["message"]["content"]
                # try:
                match_start = re.search('\[\{', text)
                match_end = re.search('\}\]', text)

                json_start = match_start.span()[0]
                json_end = match_end.span()[1]
                extraction_dict = json.loads(text[json_start:json_end])
                # except:

                with open('output/logs/log_file_{}_{}_bar.txt'.format(model_name, shot), 'a') as file:
                    file.write("{}:{}".format(tab_id, text))
                    file.write("\n")
        except:
            with open('output/logs/filtered/log_file_{}_{}_bar.txt'.format(model_name, shot), 'a') as file:
                file.write("{}:{}".format(tab_id, response["choices"]))
                file.write("\n")

        return extraction_dict

    else:
        return []


def calc_results(list_gt, list_predict, class_wise=False):
    """Calculate precision, recall and F1 score"""
    assert (len(list_gt) == len(list_predict))
    correct_preds, total_correct, total_preds = 0., 0., 0.

    correct_preds_dict = defaultdict(int)
    total_correct_dict = defaultdict(int)
    total_preds_dict = defaultdict(int)
    f1_dict = defaultdict(float)

    p, r, f1 = 0., 0., 0.

    for i in range(len(list_predict)):
        to_eval = []
        if class_wise:
            for span in list_gt[i]:
                cls = span[-1]
                if cls == 0:
                    continue
                else:
                    to_eval.append(span)
                    total_correct_dict[cls] += 1
            for span1 in list_predict[i]:
                cls = span1[-1]
                total_preds_dict[cls] += 1
                for span2 in list_gt[i]:
                    if span1 == span2:
                        correct_preds_dict[cls] += 1

        set_predict = set(list_predict[i])
        set_gt = set(to_eval)

        if len(set_gt) > 0:
            correct_preds += len(set_gt.intersection(set_predict))
            total_preds += len(list_predict[i])
            total_correct += len(to_eval)

    if class_wise:
        for k in correct_preds_dict:
            p = correct_preds_dict[k] / total_preds_dict[k] if correct_preds_dict[k] > 0 else 0
            r = correct_preds_dict[k] / total_correct_dict[k] if correct_preds_dict[k] > 0 else 0
            f1_dict[k] = 2 * p * r / (p + r) if correct_preds_dict[k] > 0 else 0

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p, r, f1, f1_dict


def main(filtered=False, model="gpt-35-turbo-16k", shot=3, random=False):
    """
    Main experiment: prepare the test tables, the few-shot examples and prompts for the LLMs
    """
    start_time = time.time()
    labels_dict = get_label_dict(filtered=filtered)

    data_path = "data/Wiki_TabNER_final_labeled.json"
    with open(data_path, 'r') as f:
        ner_tables = json.load(f)

    train_set, test_set = torch.utils.data.random_split(ner_tables, [31235, 30000], generator=generator)

    if model == "llama-2-7b":
        init_pipeline = init_llama_model(model)
    else:
        init_pipeline = []

    all_gt_annotations = []
    all_predict_annotations = []
    all_predict_annotations_conf = []
    e = 0
    for idx in test_set.indices[:600]:
        ser_correct_annot = []
        table = ner_tables[idx][0]
        tab_id = table[0]
        tableHeaders = table[4]
        table_data = table[5]
        row_labels = table[6]

        # for the current test table take the full length
        table_csv, table_matrix = tab_to_csv(tableHeaders, table_data, cut=100)
        table_prompt = """ \n {} \n Output: """.format(table_csv)

        for i in range(len(row_labels)):
            if len(row_labels[i][0]) > 0:
                if filtered:
                    ser_correct_annot.append([el for el in row_labels[i][0] if el[-1] not in [0, 1, 3, 4]])
                else:
                    ser_correct_annot.append([el for el in row_labels[i][0] if el[-1] != 0])

        if shot > 0:
            demos = generate_example(idx, train_set.indices, shot, random=random)

        else:
            demos = """
               Example table:
               Draw|Artist|First song (original artist)|Draw,Second song (original artist)|Result
               1|Mark Evans|" Rock Your Body " ( Justin Timberlake )|5|" I Don't Want to Talk About It " ( Rod Stewart )|Safe
               Output:
               [{"entity": "Mark Evans",   "type": "Person",   "cell_index": [0, 1] },
               {"entity": "Justin Timberlake",    "type": "Person",    "cell_index": [0, 2]},
               {"entity": "Rock Your Body",    "type": "Work",    "cell_index": [0, 2]}]
               """

        base_prompt = """You are an NER expert. Extract entities from the input table using the following types: {}. 
           If the type of the entity is not one of the types above, please use type: MISC. """.format(
            ', '.join('{}' for _ in labels_dict.keys()))

        instruction_prompt = """The output is a list with dictionary for every entity in the following format: 
           [{"entity": Entity, "type": Type, "cell_index": [x,y]}]. Cell index should be one list [x,y] where x is the row number and y is the column number.
           The table header has index -1, the table content with entities start from index [0,0]. \n"""

        prompt = base_prompt.format(*labels_dict.keys()) + instruction_prompt + demos + table_prompt

        # print(prompt)

        if len(ser_correct_annot) > 0:  # look for ChatGPT annotations only if we have GT
            ser_gt = [tuple(item) for sublist in ser_correct_annot for item in sublist]
            gpt_ser_annot = get_llm_responce(init_pipeline, model, shot, prompt, tab_id)

            if len(gpt_ser_annot) > 0:
                e += 1
                gpt_ser_pred, gpt_ser_conf = llm_responce(table_matrix, gpt_ser_annot, labels_dict)
                all_gt_annotations.append(ser_gt)
                all_predict_annotations.append(gpt_ser_pred)
                all_predict_annotations_conf.append(gpt_ser_conf)

            # print("Ground truth", ser_gt)
            # print("GPT predict", gpt_ser_annot)

            if e % 50 == 0:
                p, r, f1, f1_class = calc_results(all_gt_annotations, all_predict_annotations, class_wise=True)
                print("Current eval {}, precision {}, recall {} and f1 {}. Class wise res {}".format(e, p, r, f1,
                                                                                                     f1_class))
                metrics = {"num_tab": e, "precision": p, "recall": r, "f1": f1, "f1_class": f1_class}
                save_metrics(metrics, model, shot, filtered, random)

    plot_conf_matrix(model, shot, e, random, all_gt_annotations, all_predict_annotations_conf, labels_dict)
    p, r, f1, f1_class = calc_results(all_gt_annotations, all_predict_annotations, class_wise=True)
    metrics = {"num_tab": e, "precision": p, "recall": r, "f1": f1, "f1_class": f1_class}
    save_metrics(metrics, model, shot, filtered, random)
    print(time.time() - start_time)

    return p, r, f1, f1_class


if __name__ == "__main__":
    main()
