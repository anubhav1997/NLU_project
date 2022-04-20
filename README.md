# Domain Adaptation for Multilingual POS tagging
### This is a repository that contains the code for our coursework project for DSGA-1012 Natural Language Understanding named "Domain Adaptation for Multilingual POS Tagging"

This project implements a Multilingual POS tagging algorithm and adapting it for various domains. 

The folder <it>code<\it> contains Jupyter Notebooks for our implementation.

The file <it>Data_generation.ipynb</it> file that merges the datasets of different languages maps language specific tags to the universal tag set. 

The file <it>Data_cleaning.ipynb</it> is used to remove any values beling to data types other than <b>string</b> from the dataset. 

The file <it>Baselines_plus_BERT.ipynb</it> contains our implementation of each of the baseline models for our project namely:

1. Bi-LSTM model for Part-of-Speech Tagging (Plank et. al)
2. Bi-LSTM model for Part-of-Speech Tagging (Munoz et. al 2020)

The folder <it>training_data</it> contains the entire vocabulary and all the tags.<br>

The folder <it>Tag_Mappings</it> contains the mappings of the tags for our chosen languages to their corresponding universal tags.<br>



