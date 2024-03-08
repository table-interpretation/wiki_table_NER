import unittest
import json
import torch
from ner_prompting import calc_results, gpt_responce

seed_nr = 42
generator = torch.Generator().manual_seed(seed_nr)

class TestNER(unittest.TestCase):

   # def test_Completion_model(self):


    def test_calc_result(self):
        gt = [[(0, 0, 10, 17, 6), (0, 1, 0, 15, 6), (0, 1, 16, 26, 6), (0, 2, 0, 15, 6), (1, 1, 12, 22, 4)],
              [(1, 1, 18, 31, 6), (1, 2, 0, 10, 6), (1, 3, 0, 10, 6), (1, 3, 17, 31, 6), (1, 3, 22, 41, 6)],
              [(2, 1, 12, 22, 6), (2, 1, 12, 25, 6), (2, 2, 20, 33, 6), (2, 2, 15, 34, 6)]]

        gpt = [[(0, 0, 0, 17, 2), (0, 1, 0, 15, 4), (0, 3, 0, 12, 4), (0, 3, 13, 18, 3), (1, 1, 12, 22, 4)],
               [(1, 0, 0, 17, 2), (1, 1, 0, 5, 3), (1, 1, 23, 36, 4), (1, 2, 0, 10, 3), (1, 2, 17, 34, 4)],
               [(2, 2, 30, 48, 4), (2, 2, 49, 62, 4), (2, 2, 63, 82, 4)]]

        res = calc_results(gt, gpt, class_wise=True)

        self.assertEqual(res[0], 0.2)  # precision
        self.assertEqual(res[1], 1.0)  # recall

    def test_calc_parsed_output(self):
        file = "../log_file_davinci.txt"
        data_path = "../data/final_NER_labeled_dataset_correct.csv"
        with open(data_path, 'r') as f:
            ner_tables = json.load(f)
        train_set, test_set = torch.utils.data.random_split(ner_tables, [43600, 7693], generator=generator)

        labels_dict = {
            'Activity': 1,
            'Organisation': 2,
            'ArchitecturalStructure': 3,
            'Event': 4,
            'Place': 5,
            'Person': 6,
            'Work': 7,
        }
       # calc_parsed_output(file, labels_dict, ner_tables, test_set)

    def test_gpt_responce(self):

        table= [['Matthew Baugh', '"Tournament of the Treasure"', 'Steve Costigan , The Black Coats , Townsend Harper'],
                ['Nicholas Boving', '"Wings of Fear"', 'Harry Dickson , Bulldog Drummond'],
                ['Robert Darvel', '"The Man With the Double Heart"', 'The Nyctalope'],
                ['Matthew Dennion', '"The Treasure of Everlasting Life"', 'Allan Quatermain , Dr. Loveless , The Black Coats'],
                ['Win Scott Eckert', '"Violet\'s Lament"', 'Lady Blakeney , The Black Coats'],
                ['Martin Gately', '"Wolf at the Door of Time"', 'Doctor Omega , Moses Nebogipfel , The Nyctalope'],
                ['Travis Hiltz', '"What Lurks in Romney Marsh?"', 'Doctor Omega , Doctor Syn'],
                ['Paul Hugli', '"As Time Goes By..."', 'Doctor Omega , Rick Blaine'],
                ['Rick Lai', '"Gods of the Underworld"', 'The Black Coats , Vautrin'],
                ['Jean-Marc Lofficier', '"Dad"', 'Glinda of Oz']]

        json_responce =  [{"entity": "Matthew Baugh", "type": "Person", "cell_index": [0, 0]},
                        {"entity": "Tournament of the Treasure", "type": "Work", "cell_index": [0, 1]},
                        {"entity": "Steve Costigan", "type": "Person", "cell_index": [0, 2]},
                        {"entity": "The Black Coats", "type": "Group", "cell_index": [0, 2]},
                        {"entity": "Townsend Harper", "type": "Person", "cell_index": [0, 2]},
                        {"entity": "Nicholas Boving", "type": "Person", "cell_index": [1, 0]},
                        {"entity": "Wings of Fear", "type": "Work", "cell_index": [1, 1]},
                        {"entity": "Harry Dickson", "type": "Person", "cell_index": [1, 2]},
                        {"entity": "Bulldog Drummond", "type": "Person", "cell_index": [1, 2]},
                        {"entity": "Robert Darvel", "type": "Person", "cell_index": [2, 0]},
                        {"entity": "The Man With the Double Heart", "type": "Work", "cell_index": [2, 1]},
                        {"entity": "The Nyctalope", "type": "Person", "cell_index": [2, 2]},
                        {"entity": "Matthew Dennion", "type": "Person", "cell_index": [3, 0]},
                        {"entity": "The Treasure of Everlasting Life", "type": "Work", "cell_index": [3, 1]},
                        {"entity": "Allan Quatermain", "type": "Person", "cell_index": [3, 2]},
                        {"entity": "Dr. Loveless", "type": "Person", "cell_index": [3, 2]},
                        {"entity": "The Black Coats", "type": "Group", "cell_index": [3, 2]},
                        {"entity": "Win Scott Eckert", "type": "Person", "cell_index": [4, 0]},
                        {"entity": "Violet's Lament", "type": "Work", "cell_index": [4, 1]},
                        {"entity": "Lady Blakeney", "type": "Person", "cell_index": [4, 2]},
                        {"entity": "The Black Coats", "type": "Group", "cell_index": [4, 2]},
                        {"entity": "Martin Gately", "type": "Person", "cell_index": [5, 0]},
                        {"entity": "Wolf at the Door of Time", "type": "Work", "cell_index": [5, 1]},
                        {"entity": "Doctor Omega", "type": "Person", "cell_index": [5, 2]},
                        {"entity": "Moses Nebogipfel", "type": "Person", "cell_index": [5, 2]},
                        {"entity": "The Nyctalope", "type": "Person", "cell_index": [5, 2]}]

        labels_dict = {'Activity': 1,
            'Organisation': 2,
            'ArchitecturalStructure': 3,
            'Event': 4,
            'Place': 5,
            'Person': 6,
            'Work': 7 }

        responce = gpt_responce(table, json_responce, labels_dict)

        print(responce)



if __name__ == '__main__':
    unittest.main()
