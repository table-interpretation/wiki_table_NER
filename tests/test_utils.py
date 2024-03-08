import unittest
import json
import torch
from utils import tab_to_csv, parse_output, get_label_dict, get_correct_anno, generate_example, plot_conf_matrix
from gpt_utils import *

seed_nr = 42
generator = torch.Generator().manual_seed(seed_nr)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        data_path = "../data/final_NER_labeled_dataset.csv"
        with open(data_path, 'r') as f:
            self.ner_tables = json.load(f)

        self.train_set, self.test_set = torch.utils.data.random_split(self.ner_tables, [49271, 2000], generator=generator)
        self.labels_dict = get_label_dict(filtered=False)

    def test_parsing(self):
        file = "../log_file_davinci.txt"
        raw_output, new_output = parse_output(file)
        print(len(raw_output.keys()))
        print(len(new_output.keys()))

    def test_table_transformation(self):
        train_set, test_set = torch.utils.data.random_split(self.ner_tables, [49271, 2000], generator=generator)

        for idx in test_set.indices[:3]:
            ser_correct_annot = []
            gpt_ser_pred = []

            table = self.ner_tables[idx][0]

            tab_id = table[0]
            tableHeaders = table[4]
            table_data = table[5]
            row_labels = table[6]

            table_csv, table_matrix = tab_to_csv(tableHeaders, table_data)
            print(table_csv)
            print(table_matrix)

    def test_read_gpt_configs(self):
        conf = get_gpt_configuration(
            "https://openai-aiattack-msa-000898-eastus-smrattack-00.openai.azure.com/",
            "gpt-35-turbo-instruct",
        )
        print(conf)
        self.assertTrue(conf)

    def test_get_labels(self):

        # 4 types: Organization, Place, Person and Work
        filtered_labels_dict = get_label_dict(filtered=True)
        labels_dict = get_label_dict(filtered=False)

        self.assertEqual(len(filtered_labels_dict), 4)
        self.assertEqual(len(labels_dict), 7)

    def test_tab_to_csv(self):

        table = self.ner_tables[371][0]

        tableHeaders = table[4]
        table_data = table[5]
        tab_csv, tab_matrix = tab_to_csv(tableHeaders, table_data, sep = '|', cut = 4)

        print(tab_csv)
        print(tab_matrix)

    def test_get_correct_anno(self):

        labels_dict_rev = {v: k for k, v in self.labels_dict.items()}

        test_csv, test_output = get_correct_anno(self.ner_tables[45445][0], labels_dict_rev)

        print(test_csv)
        print(test_output)

    def test_generate_example(self):

        idx = 42886
        k = 1
        examples = generate_example(idx, self.train_set.indices, k)
        print(examples)

    def test_plot_conf_matx(self):

        with open("../output/all_gt.json") as f:
            all_gt = json.load(f)

        with open("../output/all_predict.json") as f:
            all_predict = json.load(f)
        plot_conf_matrix('test', 1, 10, all_gt, all_predict, self.labels_dict)


if __name__ == '__main__':
    unittest.main()
