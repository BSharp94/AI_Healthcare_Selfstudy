# AI In Healthcare - Self Study on Knowledge Graphs


In this project we explore one approach for using LLM to assist in the development of medical knowledge graphs.

## Dataset

We will use a medical texts dataset that is publicly available in kaggle. 

(Kaggke Link to Dataset)[https://www.kaggle.com/datasets/chaitanyakck/medical-text]

The dataset contains the medical abstracts that descibe the current conditions of patients. These abstracts give quick overviews of the patient's primary illness, any concerning symptoms, and the tests and procedures they are using to treat the patient.

In order to run the program, create a folder called 'dataset' at the root level. Download the archive.zip file from the url and place it in the folder. Extract the context of the zip file.

This should created two new files:

- ./dataset/archive/test.dat
- ./dataset/archive/train.dat

## Run the code

To run the code, first ensure you have the dependency libraries installed. (requirements.txt).

Next run the following script:


```
python extract.py
```

The initial run may take several minutes to download the necessary model assets. 

