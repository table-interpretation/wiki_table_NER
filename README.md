# Wiki-TabNER

This repository contains the dataset and the code for the paper [Wiki-TabNER:Advancing Table Interpretation Through Named
Entity Recognition](https://arxiv.org/abs/2403.04577).

Motivated by the simpler tables commonly used for table interpretation tasks, we propose a new dataset annotated with NERs within tables. 
![example table](./notebooks/plots/motivation.png) 
Here is an example table how the tables in the Wiki-TabNER dataset are annotated. 

For the annotations of the entities we used the existing links between WIkipedia pages and Dbpedia entities.
We used the following entities types: 
<em> Activity, Organisation, Architectural Structure, Event, Place, Person and Work</em>. 


Plots describing the dataset van be found in the [plots folder](./notebooks/plots).
The dataset contains 51 271 tables annotated entities with BIO-labels and span-based labels. 
The labels have been extracted from Dbpedia and Yago. It can be downloaded from [here](https://zenodo.org/records/10794526).
Here, we also provide the file with the ground truth needed for the evaluation of the entity linking task.

We provide evaluation of the current LLMs (GPT 3.5, GPT 4 and Llama2) for evaluation of the NER in tables task.
The evaluation of the LLMs is in the [ner_prompting.py](ner_prompting.py) file. In order to run the evaluation of the Open-AI models, 
a configuration of the parameters is required. 

If you use the Wiki-TabNER dataset, please cite the related paper:
```
@misc{koleva2024wikitabneradvancing,
      title={Wiki-TabNER:Advancing Table Interpretation Through Named Entity Recognition}, 
      author={Aneta Koleva and Martin Ringsquandl and Ahmed Hatem and Thomas Runkler and Volker Tresp},
      year={2024},
      eprint={2403.04577},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```