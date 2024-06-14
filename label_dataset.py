import pandas as pd
import json
import numpy as np
import re
from tqdm import tqdm

from utils import add_mention


def main():
    """
    Script for labeling and transforming the tables in the dataset. The tables are saved in json format, where each
    table has 7 fields:
    tab_id - the table ID
    pg_Title - the title of the Wikipedia page
    section_Title - the section from where the table is
    table_caption - title of the table
    tableHeaders - the column names
    table_data - the data in the tables where every cell is represented as a list [cell index, cell text]
    row_labels - the span-based labels for every linked entity. [cell index, start, end of span, label]
    """

    dataset_entities = pd.read_csv("data/labeled_entities/full_dataset_entities_labeled_dbp_yago_final.csv")
    dataset_entities.drop('Unnamed: 0', axis=1, inplace=True)


    nclass = dataset_entities['class'].to_numpy()
    nwikidata_id = dataset_entities['wikidata_id'].to_numpy()
    nentity = dataset_entities['entity'].to_numpy()
    nlabels = list(zip(nclass, nwikidata_id))

    dataset_entities['mention'] = None
    dataset_entities['table_id'] = None

    entity_index = {entity: index for index, entity in dataset_entities['entity'].items()}

    labels_dict = {"None": 0,
                   "Activity": 1,
                   "Organisation": 2,
                   "ArchitecturalStructure": 3,
                   "Event": 4,
                   "Place": 5,
                   "Person": 6,
                   "Work": 7,
                   }

    # load the extracted tables
    data_path = "data/wiki_tabner_original_clean.json"
    dataset_df = pd.read_json(data_path, lines=True)
    weird_entities = {}
    all_tables = []
    with tqdm(total=len(dataset_df)) as progress:
        with pd.read_json(data_path, lines=True, chunksize=5000) as reader:
            for chunk in reader:
                for index, t in chunk.iterrows():

                    t = dataset_df.iloc[index]

                    new_table = []
                    tableHeaders_cleaned = []
                    tab_id = t["_id"]

                    pg_Title = t["pgTitle"]
                    section_Title = t["sectionTitle"]
                    table_caption = t["tableCaption"]
                    tableHeaders = t["tableHeaders"]
                    table = t["tableData"]

                    headers_cells = [h for h in tableHeaders][0]
                    for c, head in enumerate(headers_cells):
                        tableHeaders_cleaned.append([[-1, c], head['text']])

                        table_data = []
                        row_labels = []
                        for i, row in enumerate(table):
                            for j, cell in enumerate(row):

                                per_cell_labels = []
                                cell_spans = []
                                table_data.append([[i, j], cell["text"]])
                                if len(cell['surfaceLinks']) > 0:
                                    for k in range(len(cell['surfaceLinks'])):
                                        subcell_i = cell['surfaceLinks'][k]

                                        start_idx = subcell_i["offset"]
                                        end_idx = subcell_i["endOffset"]
                                        ent_text = subcell_i["surface"]
                                        text_lookup = subcell_i["target"]["title"]

                                        try:
                                            span_search = re.search(ent_text, cell["text"])
                                            if span_search:
                                                token_start = span_search.span()[0]
                                                token_end = span_search.span()[1]
                                        except:
                                            token_start = start_idx
                                            token_end = end_idx
                                            weird_entities[ent_text] = (text_lookup, tab_id)

                                        lookup_idx = np.where(nentity == text_lookup)[0]
                                        ent_label = nlabels[lookup_idx[0]] if lookup_idx.any() else (None, None)

                                        if ent_label[1]:
                                            add_mention(ent_text, tab_id, text_lookup, dataset_entities, entity_index)

                                        if ent_label[0] in labels_dict:
                                            span_based_annotation = (
                                                i, j, token_start, token_end, labels_dict[ent_label[0]])
                                        else:
                                            span_based_annotation = (i, j, token_start, token_end, 0)

                                        cell_spans.append(span_based_annotation)

                                    cell["subcell_labels"] = cell_spans
                                    per_cell_labels.append(cell_spans)
                                # per_cell_labels.append(mapped)

                                if len(per_cell_labels) > 0:
                                    row_labels.append(per_cell_labels)

                    new_table.append([tab_id,
                                      pg_Title,
                                      section_Title,
                                      table_caption,
                                      tableHeaders_cleaned,
                                      table_data,
                                      row_labels]
                                     )
                    all_tables.append(new_table)
                    progress.update()

    print(len(all_tables))
    with open('data/Wiki_TabNER_final.json', 'w') as f:
        json.dump(all_tables, f)

    dataset_entities.to_csv("data/labeled_entities/dataset_entities_labeled_linked.csv")


if __name__ == "__main__":
    main()
