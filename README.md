# Wiki-TabNER

This repository contains the dataset and the code for the paper Wiki-TabNER:Integrating NER in Wikipedia tables. 
The dataset can be downloaded from [here](https://zenodo.org/records/11654240).

Motivated by the lack of more complex tables in the existing datasets commonly used for table interpretation tasks, we propose a new dataset annotated with NERs within tables. 
![example table](./notebooks/plots/motivation.png) 
Here is an example table how the tables in the Wiki-TabNER dataset are annotated. 

### Extraction of tables

The tables in the Wiki_TabNER dataset are extracted from the [WikiTables corpus](http://websail-fe.cs.northwestern.edu/TabEL/).
We extracted tables which have on average two linked entities per cell. 
The extraction of the tables is detailed in the [1.Extract_TabNER_tables notebook](./notebooks/1.Extract_TabNER_tables.ipynb).

### Labeling the entities
In the WikiTables corpus the tables are represented in json structure. For every cell, there is information for the start, end of the linked entities,
their Wikipedia page and their _surface text_.
We use the _surface text_ for the entities to match them to DBpedia labeled instances.
We add the labels of the entities at the end of the table structure, where every labeled entity is represeted
with a list [row index, col index, start span, end span, label].

The labeling of the entities is detailed in the [2.Extract_TabNER_entities notebook](./notebooks/2.Extract_TabNER_entities.ipynb).

We used the following entities types for labeling: 
<em> Activity, Organisation, Architectural Structure, Event, Place, Person and Work</em>. 
We provide the labeled dataset and the labeled named entities, linked to their Wikidata IDs in a separate file which can be downloaded [here](https://zenodo.org/records/11654240).


### Evaluation of LLMs on within tables NER
We provide evaluation of the current LLMs (GPT 3.5, GPT 4 and Llama2) for evaluation of the NER in tables task.
The evaluation of the LLMs is in the [ner_prompting.py](ner_prompting.py) file. In order to run the evaluation of the Open-AI models, 
a configuration of the parameters is required. 



