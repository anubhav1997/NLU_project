# Domain Adaptation for Multilingual POS tagging
### This is a repository that contains the code for our coursework project for DSGA-1012 Natural Language Understanding named "Domain Adaptation for Multilingual POS Tagging"

This project implements Multilingual POS tagging and its adaptation for various domains. 

The folder ```code``` contains Jupyter Notebooks for our implementation.

The file ```Data_generation.ipynb``` file that merges the datasets of different languages maps language specific tags to the universal tag set. 

The file ```Data_cleaning.ipynb``` is used to remove any values beling to data types other than <b>string</b> from the dataset. 

The file ```Baselines_plus_BERT.ipynb``` contains our implementation of each of the baseline models for our project namely:

1. Bi-LSTM model for Part-of-Speech Tagging (Plank et. al)
2. RNN model for Part-of-Speech Tagging (Munoz et. al 2020)

The folder ```training_data``` contains the entire vocabulary and all the tags.<br>

The folder ```Universal_Tag_Mappings``` contains the mappings of the tags for our chosen languages to their corresponding universal tags.<br>
  
The folder ```multi_domain``` contains the data from sources belonging to various domains and in different languages. Currently, we are using data from the social media domain in the German language (A harmonised testsuite for social media POS tagging (DE)) and clinical data in English (GENIA Corpus) and Spanish (Spanish Clinical Case Corpus) languages to test the cross-lingual domain adaptation capabilities of our model.
  
The folder ```parsers``` contains the code that was used to parse and organize the files in the ```multi_domain``` folder. 



