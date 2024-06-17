# Wiki-TabNER

This repository contains the dataset and the code for the paper **Wiki-TabNER:Integrating NER in Wikipedia tables**. 
The dataset can be downloaded from [here](https://zenodo.org/records/11654240).

Motivated by the lack of more complex tables in the existing datasets commonly used for table interpretation tasks, we propose a new dataset annotated with NERs within tables. 

## Example table

--- 

Here is an example table how the tables in the Wiki-TabNER dataset are annotated. 
![example table](./notebooks/plots/motivation.png) 

The raw table representation in the WikiTable corpus contains information about the linked entities in the cells.
Here is how the table from the example is shown in the original corpus. <br /> 
After the text from the cells, *'tdHtmlString'* lists all the links of the linked entities within this cell. <br />  The *'surfaceLinks'* indicate the 
start and end of the linked entity, the title in Wikipedia and the surface text of the entity. <br /> 
```
'_id': '17499143-1',
'pgTitle': 'Angela Maxwell' ,
'sectionTitle': 'Programs',
'tableCaption': 'Programs',
'tableHeaders': [ {'text': 'Season', ...},
                  {'text': 'Short Program', ...},
                  {'text': 'Free Skating', ...},
                  {'text': 'Exhibition', ...},
                  {'text': '2009-2010', ...}]
'tableData': [[{'text': '2009-2010', 'tdHtmlString':'2009-2010 </th>', 'surfaceLinks': []},
{'text': 'Santa Maria (del Buen Ayre) by Gotan Project Libertango by Ástor Piazzolla',
 'tdHtmlString': '<a href="http://www.wikipedia.org/wiki/La_Revancha_del_Tango" shape="rect">Santa Maria (del Buen Ayre)</a> by
                  <a href="http://www.wikipedia.org/wiki/Gotan_Project" shape="rect">Gotan Project</a>
                  <a href="http://www.wikipedia.org/wiki/Libertango" shape="rect">Libertango</a> by
                  <a href="http://www.wikipedia.org/wiki/Ástor_Piazzolla" shape="rect">Ástor Piazzolla</a> </td>',
 'surfaceLinks': [{'offset': 0, 'endOffset': 1,  'target': {'id': 4683862, 'language': 'en', 'title': 'La_Revancha_del_Tango'}, 'surface': 'Santa Maria (del Buen Ayre)'},
                  {'offset': 31, 'endOffset': 44,  'target': {'id': 1054664, 'language': 'en', 'title': 'Gotan_Project', 'redirecting': False, 'namesapce': 0}, 'surface': 'Gotan Project'},
                  {'offset': 45, 'endOffset': 55,  'target': {'id': 18223717, 'language': 'en', 'title': 'Libertango', 'redirecting': False, 'namesapce': 0}, 'surface': 'Libertango'},
                  {'offset': 15, 'endOffset': 30,  'target': {'id': 44903, 'language': 'en', 'title': 'Ástor_Piazzolla', 'redirecting': False, 'namesapce': 0}, 'surface': 'Ástor Piazzolla'}],
                }

```
Below we show the transformed table in Wiki_TabNER. Note that in the WikiTables corpus, the *'offset'* and *'endOffset'* positions of the entities are not always correct (for ex for the first and last entity in the cell).
When adding the annotations for the named entities, we make sure to look for the position of the *'surface'* text of the entity in the cell text and we add these positions as the start and end of the named entity.
```
['17499143-1',
  'Angela Maxwell',
  'Programs',
  'Programs',
  [[[-1, 0], 'Season'],
   [[-1, 1], 'Short Program'],
   [[-1, 2], 'Free Skating'],
   [[-1, 3], 'Exhibition']],
  [[[0, 0], '2009-2010'],
   [[0, 1], 'Santa Maria (del Buen Ayre) by Gotan Project Libertango by Ástor Piazzolla'],
   [[0, 2], 'Nostradamus by Maksim Mrvica Vampire Knight Guilty from Vampire Knight soundtrack by Haketa Takefumi'],
   [[0, 3], ''],
   ...
   ]
  [[0, 1, 0, 26, 7],
   [0, 1, 31, 44, 2],
   [0, 1, 45, 55, 7],
   [0, 1, 59, 74, 0],
  [[0, 2, 15, 28, 2], [0, 2, 29, 43, 7]]]
  ...
  ]
```


## Extraction of tables

---


The tables in the Wiki_TabNER dataset are extracted from the [WikiTables corpus](http://websail-fe.cs.northwestern.edu/TabEL/).
We extracted tables which have on average two linked entities per cell. 
The extraction of the tables is detailed in the [1.Extract_TabNER_tables notebook](./notebooks/1.Extract_TabNER_tables.ipynb).

## Labeling the entities

---

We use the _surface text_ for the entities to match them to DBpedia labeled instances.
As shown in the example above, we add the labels of the entities at the end of the table structure, where every labeled entity is represented
with a list [row index, col index, start span, end span, label].

The labeling of the entities is detailed in the [2.Extract_TabNER_entities notebook](./notebooks/2.Extract_TabNER_entities.ipynb).

We used the following entities types for labeling: 
<em> Activity, Organisation, Architectural Structure, Event, Place, Person and Work</em>. 
We provide the labeled dataset and the labeled named entities, linked to their Wikidata IDs in a separate file which can be downloaded [here](https://zenodo.org/records/11654240).



### Evaluation of LLMs on within tables NER
We provide evaluation of the current LLMs (GPT 3.5, GPT 4 and Llama2) for evaluation of the NER in tables task.
The evaluation of the LLMs is in the [ner_prompting.py](ner_prompting.py) file. In order to run the evaluation of the Open-AI models, 
a configuration of the parameters is required. 



