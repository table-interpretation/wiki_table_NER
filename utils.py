import pandas as pd
import os
import csv
import json
import re
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_metrics(metrics, model, shot, filtered, random):
    """
    Saving the results during the experiments.
    """
    if filtered:
        csv_filename = "filtered/results_{}_{}_bar_{}.csv".format(model, shot, random)
    else:
        csv_filename = "results_{}_{}_bar_{}.csv".format(model, shot, random)

    flattened_metrics = {'num_tab': metrics['num_tab'], 'precision': metrics['precision'], 'recall': metrics['recall'],
                         'f1': metrics['f1']}
    flattened_metrics.update(metrics['f1_class'])

    try:
        with open(os.path.join('output', csv_filename), 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            existing_columns = reader.fieldnames
    except FileNotFoundError:
        existing_columns = None

    with open(os.path.join('output', csv_filename), 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['num_tab', 'precision', 'recall', 'f1', 1, 2, 3, 4, 5, 6, 7])

        # Write header if the file is newly created
        if existing_columns is None:
            writer.writeheader()
        # Write the metrics
        writer.writerow(flattened_metrics)


def plot_conf_matrix(model_name, shot, e, random, list_gt, list_predict, labels_dict):
    """
    Plotting the confusion matrix.
    """
    true_label = []
    predicted_label = []

    labels_dict["MISC"] = 10

    for i in range(len(list_predict)):
        for el in list_predict[i]:
            for tup in list_gt[i]:
                if el[0:4] == tup[0:4]:
                    true_label.append(tup[4])
                    predicted_label.append(el[4])

    # This is because the MISC label is 0 and it messes up the plot.
    # We change it to 10, so that the MISC labels are shown last.
    for i, el in enumerate(predicted_label):
        if el == 0:
            predicted_label[i] = 10

    cm = confusion_matrix(true_label, predicted_label)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_dict.keys(), yticklabels=labels_dict.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.savefig("output/plots/cm_{}_{}_{}_bar_{}.png".format(e, model_name, shot, random))
    # plt.show()


def get_label_dict(file_path="data/labels_dict.txt", filtered=False):
    """
    Loading the labels from a file. IF filtered, we remove Activity, ArchitecturalStructure and Event.
    """
    labels_dict = {
        'Activity': 1,
        'Organisation': 2,
        'ArchitecturalStructure': 3,
        'Event': 4,
        'Place': 5,
        'Person': 6,
        'Work': 7,
    }

    labels_dict_filtered = {
        'Organisation': 2,
        'Place': 5,
        'Person': 6,
        'Work': 7,
    }

    if filtered:
        return labels_dict_filtered
    else:
        return labels_dict


def tab_to_csv(tableHeaders, table_data, sep='|', cut=30):
    """
    Converting the table content into csv format | delimited.
    """
    table_contents = ""
    original_table_dict = {}

    columns = [tableHeaders[i][1] for i in range(len(tableHeaders))]

    table_contents = """Table:\n"""
    table_contents += sep.join([str(c) for c in columns])
    table_contents += '\n'

    for cell_info, content in table_data:
        row, col = cell_info
        original_table_dict.setdefault(row, {})[col] = content

    if cut < len((original_table_dict)):
        table_dict = {k: original_table_dict[k] for k in list(original_table_dict)[:cut]}
    else:
        table_dict = original_table_dict

    num_rows = max(table_dict.keys()) + 1
    num_cols = max(max(row.keys()) for row in table_dict.values()) + 1

    table_matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]

    for row, cols in table_dict.items():
        for col, content in cols.items():
            table_matrix[row][col] = content

    for i in range(len(table_matrix)):
        table_contents += sep.join([str(c) for c in table_matrix[i]])
        table_contents += '\n'

    return table_contents, table_matrix


def process_single_table_gpt(model, tokenizer, table):
    """
    Used for generating table embeddings of a tables.
    Returns the mean of the token representations - one vector with shape (768,)
    """

    pgTitle = table[1]
    secTitle = table[2]
    tableHeaders = table[4]

    table_data = table[5]

    headers = ' '.join([tableHeaders[i][1] for i in range(len(tableHeaders))])
    table_rows = ' '.join([row[1] for row in table_data])

    table_text = str(pgTitle) + str(secTitle) + headers + table_rows
    tokenized_table = tokenizer(table_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        model_output = model(**tokenized_table)

    embeddings = model_output.last_hidden_state
    table_representation = torch.mean(embeddings, dim=1).squeeze()

    return table_representation


def get_correct_anno(table, labels_dict_rev):
    """
    For a selected table, extract the first 4 rows, the annotations and the table in matrix format.
    The First 4 rows and annotations are used for generating prompt examples.
    The table in matrix format is used during the evaluation of the output.
    """
    tab_id = table[0]
    tableHeaders = table[4]
    table_data = table[5]
    row_labels = table[6]
    ser_correct_annot = []

    table_csv, table_matrix = tab_to_csv(tableHeaders, table_data, sep='|', cut=400)

    for i in range(len(row_labels)):
        if len(row_labels[i][0]) > 0:
            ser_correct_annot.append(row_labels[i][0])

    outputs = []
    for anno in ser_correct_annot:
        for a in anno:
            if a[-1] != 0 and a[0] < 4:  # take the annotations for the first 4 rows only
                # print(a)
                cell_index = [a[0], a[1]]
                type_ = labels_dict_rev[a[-1]]
                text = table_matrix[a[0]][a[1]][a[2]:a[3]]

                outputs.append({"entity": text, "type": type_, "cell_index": cell_index})

    return table_csv, outputs, table_matrix


def generate_example(idx, train_set_indices, k, random=False):
    """
    Used for generating few-shot examples.
    The idx is the index of the current test table in the dataset.
    Load the generated table embeddings and find the k most similar to the current test table
    Using the indices of the most similar tables, retrieve their annotated NERs from the saved file.
    The generation of the embeddings and the NERs for the tables is done in the notebook generate_tab_embeddings.
    """

    all_table_embeddings = np.load('output/bert_all_table_embeddings.npy')
    train_embeddings = all_table_embeddings[train_set_indices]
    similar = np.dot(all_table_embeddings[idx], train_embeddings.T)
    topk_similar_indices = np.argsort(similar, axis=0)[-k:]

    with open("output/similar_tables_examples.json", "r") as f:
        similar_tables = json.load(f)

    if k == 1:
        if random == False:
            i = topk_similar_indices[0]
            d = similar_tables[str(train_set_indices[i])]
        else:
            i = '10877'
            d = similar_tables[i]
        demos = d[0] + "Output: " + json.dumps(d[1], ensure_ascii=False) + "\n"

    else:
        demos = ""
        if random == False:
            for i in topk_similar_indices:
                d = similar_tables[str(train_set_indices[i])]
                demo = d[0] + "Output: " + json.dumps(d[1], ensure_ascii=False) + "\n"
                demos += demo
        else:
            for i in ['9266', '35188', '10877']:
                d = similar_tables[i]
                demo = d[0] + "Output: " + json.dumps(d[1], ensure_ascii=False) + "\n"
                demos += demo

    return demos


def eval_llama_output(ner_tables, test_set_indices, parsed_output_llama, labels_dict, filtered):
    """
    Load parsed output from LLama for evaluation. Used in the notebook Llama_model.
    """
    if filtered:
        del labels_dict['Activity']
        del labels_dict['ArchitecturalStructure']
        del labels_dict['Event']

    labels_dict_rev = {v: k for k, v in labels_dict.items()}
    dict_results = {"true_label": [], "predicted_label": [], "incorrect_span_predict": [], "incorrect_span_gt": [],
                    "incorrect_cell_predict": [], "incorrect_cell_gt": []}
    total_preds = 0
    all_gt_annotations = []
    all_predict_annotations = []
    all_predict_conf = []

    for key in parsed_output_llama.keys():
        ser_correct_annot = []
        for idx in test_set_indices:
            if ner_tables[idx][0][0] == key:
                table = ner_tables[idx][0]
                row_labels = table[6]

                for i in range(len(row_labels)):
                    if len(row_labels[i][0]) > 0:
                        if filtered:
                            ser_correct_annot.append([el for el in row_labels[i][0] if el[-1] not in [0, 1, 3, 4]])
                        else:
                            ser_correct_annot.append([el for el in row_labels[i][0] if el[-1] != 0])

                ser_gt = [tuple(item) for sublist in ser_correct_annot for item in sublist]
                all_gt_annotations.append(ser_gt)

                tab_csv, outputs, tab_matrix = get_correct_anno(ner_tables[idx][0], labels_dict_rev)
                llama_predicted, llama_conf = llm_responce(tab_matrix, parsed_output_llama[key], labels_dict)

                all_predict_annotations.append(set(llama_predicted))
                all_predict_conf.append((set(llama_conf)))
                total_preds += len(set(llama_predicted))

                for el in llama_predicted:
                    for tup in ser_gt:

                        if el[0:4] == tup[0:4] and el[4] != tup[4]:
                            dict_results["true_label"].append(tup[4])
                            dict_results["predicted_label"].append(el[4])

                            # for span - check if cell index and label are the same
                        elif ((el[0:3] == tup[0:3] and el[3] != tup[3] and el[4] == tup[4])
                              or (el[0:2] == tup[0:2] and el[2] != tup[2] and el[3:] == tup[3:])):
                            dict_results["incorrect_span_predict"].append(el)
                            dict_results["incorrect_span_gt"].append(tup)

                        # for cell - check if only the column index is different
                        elif ((el[0] == tup[0] and el[1] != tup[1] and el[2:] == tup[2:])
                              or (el[0] != tup[0] and el[1:] == tup[1:])):
                            dict_results["incorrect_cell_predict"].append(el)
                            dict_results["incorrect_cell_gt"].append(tup)

    dict_results["total_preds"] = total_preds
    dict_results["all_gt_annotations"] = all_gt_annotations
    dict_results["all_predict_annotations"] = all_predict_annotations
    dict_results["all_predict_conf"] = all_predict_conf  # predictions for the conf matrix contain labels MISC

    return dict_results


def llm_responce(table_matrix, json_responce, labels_dict):
    """
    Function that transforms the LLMs responce into coorect format for the span-based evaluation.
    Input is the table in matrix format so that the rows and cols are correctly indexed.
    Output is list with tuples (row_idx, col_idx, start_span, end_span, label)
    """
    gpt_ners = []
    gpt_ners_conf = []

    for entity in json_responce:

        try:
            entity_text = entity['entity']
            row_id = entity['cell_index'][0]
            col_id = entity['cell_index'][1]

            try:
                label = labels_dict[entity['type']]
            except:
                label = 0

            # print("Entity {}, row {}, col {}, label {}".format(entity_text, row_id, col_id, label ))

            cell_text = table_matrix[row_id][col_id]
            span_search = re.search(entity_text, cell_text)
            # print("Entity", entity_text, entity['cell_index'], "Cell", cell_text)
            token_start = span_search.span()[0]
            token_end = span_search.span()[1]

            # for calc of f1 score append only non 0 ners
            if label != 0:
                gpt_ners.append((row_id, col_id, token_start, token_end, label))

            # for calc of conf matrix, append all ners
            gpt_ners_conf.append((row_id, col_id, token_start, token_end, label))

        except Exception as e:
            # print(entity)
            continue

    return gpt_ners, gpt_ners_conf

def parse_output(model_name, file_name):
    """
    Parsing the generated output from the LLM and extracting labeled entities.
    Model name - Llama or GPT to find the subfolder with the files - this is mainly used for LLama
    full file name - the name of the experiment that generated the file
    """

    with open(os.path.join('../output/{}'.format(model_name), file_name), 'r') as file:
        output = file.read()

    pattern_text = r'Output:(.*?)(?=Output:|\Z)'
    pattern_split = r'^(\d+-\d+):(.*)$'

    # first load the parsed output into dict
    dict_output = {}

    for line in output.split('\n'):
        all_output_ents = []
        parsed_entity = []
        match = re.match(pattern_split, line.strip())
        if match:
            current_id = match.group(1)
            current_text = match.group(2)

            matches_text = re.findall(pattern_text, current_text, re.DOTALL)

            for i in range(0, len(matches_text)):
                parsed = parse_entities(matches_text[i])
                all_output_ents.append(parsed)

                # max_ents = max(all_output_ents, key = len)
            expanded = [item for sublist in all_output_ents for item in sublist]
            dict_output[current_id] = expanded
            # print(parse_entities(matches_text[0]))

    return dict_output


def parse_entities(entity_text):
    """
    Cleaning the entities from the parsed output and adding them into json.
    """
    entities = []

    # Find all JSON objects in the entity text using regular expressions
    entity_text = entity_text.replace('\\n', '')
    entity_text = entity_text.replace('\\', '')
    entity_text = entity_text.replace(']]', ']')
    entity_texts = re.findall(r'\{.*?\}', entity_text)

    for entity_text in entity_texts:
        try:
            entity = json.loads(entity_text)  # Parse the JSON object
            entities.append(entity)
        except:
            # entity = json.loads(entity_text)
            print(entity_text)

    return entities


def clean_answer(output_string):
    """
    Attempt to clean the output from GPT. Not used at the moment.
    """
    list_match = re.search(r"\[\[.*?\]\]", output_string, re.DOTALL)

    if list_match:
        list_string = list_match.group()
        list_items = list_string.strip(" []").split("], [")

        cleaned_list = []
        for item in list_items:
            token_label_pair = item.strip("[] ").split(", ")
            cleaned_token_label = [token_label.strip(" '") for token_label in token_label_pair]
            cleaned_list.append(cleaned_token_label)

        return (cleaned_list)
    else:
        print("No valid list found in the string.")
        output_index = output_string.find("Output:")

        if output_index != -1:
            next_string_index = output_string.find("\n", output_index)
            if next_string_index != -1:
                output_substring = output_string[output_index + len("Output:"):next_string_index]

                output_lines = output_substring.strip().split('\n')
                output_list = [line.split() for line in output_lines]
                print("After parsing", output_list)
                return output_list
        else:
            print(output_string)
        return []


def convert_to_iob(tokens, surfaces, labels):
    """
    Used during the labeling of the dataset.
    Generates and adds the BIO labels to the entities.
    """
    mapped = np.empty(len(tokens), dtype=object)
    mapped[:] = [(tokens[t], "O") for t in range(len(tokens))]
    i = 0
    e = 0
    t = 0
    Begin = False

    while t < len(mapped) and e < len(surfaces):

        if Begin == False and tokens[t] == surfaces[e][0]:
            Begin = True
            mapped[t] = (tokens[t], "B-" + labels[e][0]) if labels[e][0] != None else (tokens[t], "O")
            i += 1
            t += 1
        elif Begin == True and i <= len(surfaces[e]) - 1:
            mapped[t] = (tokens[t], "I-" + labels[e][0]) if labels[e][0] != None else (tokens[t], "O")
            i += 1
            t += 1

        elif Begin == True and i > len(surfaces[e]) - 1:
            Begin = False
            e += 1
            i = 0
        else:
            i = 0
            t += 1

    return mapped


def add_mention(ent_text, tab_id, text_lookup, dataset_entities):
    """
    Used during the labeling of the dataset.
    # For every labeled entity save in which table it was mentioned and what is its label
    """

    row_index = dataset_entities.index[dataset_entities['entity'] == text_lookup]
    if not row_index.empty:
        dataset_entities.loc[row_index, ['mention', 'table_id']] = [ent_text, tab_id]

    return dataset_entities
