
# Nitro-NLP: Satire Detection in Romanian News

## Introduction

This project focuses on detecting satirical content in Romanian news articles using a fine-tuned NLP model. The model was developed during the Nitro NLP hackathon and trained on a custom dataset containing both satirical and non-satirical news articles. The goal was to create a reliable classifier that could distinguish between the two types of content with high accuracy. The project highlights the importance of proper text preprocessing, including tokenization and dataset cleaning, to ensure the model's performance and generalization capabilities.

The primary goal of this project was to fine-tune a pre-existing NLP model to classify Romanian news articles as either satirical or non-satirical. After evaluating several models, I selected one that demonstrated strong performance on similar text classification tasks. The dataset underwent a thorough preprocessing pipeline, including tokenization and cleaning, which were crucial steps to ensure the model's accuracy and robustness. Proper preprocessing is essential in NLP tasks, as it directly impacts the model's ability to understand and generalize from the input data. The fine-tuning process was tailored to the specific nuances of the Romanian language, ensuring optimal performance on the task at hand.

## bert-base-romanian-cased-v1

### How to use

```python
from transformers import AutoTokenizer, AutoModel
import torch
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

Remember to always sanitize your text! Replace ``s`` and ``t`` cedilla-letters to comma-letters with :
```
text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
```
because the model was **NOT** trained on cedilla ``s`` and ``t``s. If you don't, you will have decreased performance due to ``<UNK>``s and increased number of tokens per word. 

### Evaluation

Evaluation is performed on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) UPOS, XPOS and LAS, and on a NER task based on [RONEC](https://github.com/dumitrescustefan/ronec). Details, as well as more in-depth tests not shown here, are given in the dedicated [evaluation page](https://github.com/dumitrescustefan/Romanian-Transformers/tree/master/evaluation/README.md). 

The baseline is the [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) model ``bert-base-multilingual-(un)cased``, as at the time of writing it was the only available BERT model that works on Romanian.

| Model                          |  UPOS |  XPOS  |  NER  |  LAS  |
|--------------------------------|:-----:|:------:|:-----:|:-----:|
| bert-base-multilingual-cased   | 97.87 |  96.16 | 84.13 | 88.04 |
| bert-base-romanian-cased-v1    | **98.00** |  **96.46** | **85.88** | **89.69** |

### Corpus 

The model is trained on the following corpora (stats in the table below are after cleaning):

| Corpus    	| Lines(M) 	| Words(M) 	| Chars(B) 	| Size(GB) 	|
|-----------|:--------:|:--------:|:--------:|:--------:|
| OPUS      	|   55.05  	|  635.04  	|   4.045  	|    3.8   	|
| OSCAR     	|   33.56  	|  1725.82 	|  11.411  	|    11    	|
| Wikipedia 	|   1.54   	|   60.47  	|   0.411  	|    0.4   	|
| **Total**     	|   **90.15**  	|  **2421.33** 	|  **15.867**  	|   **15.2**   	|

### Citation



```
Stefan Dumitrescu, Andrei-Marius Avram, and Sampo Pyysalo. 2020. The birth of Romanian BERT. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4324–4328, Online. Association for Computational Linguistics.
```
