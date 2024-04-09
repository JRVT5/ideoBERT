# **Final Project Proposal - Political Bias using BERT Text Classification Model**

## Group Members

[Jonah Roberts]
[David Byrne]

## Problem Statement

1. How this problem is related to the class?
Throughout the class we have dived into the different biases in human languages and the current/future capabilities for machine learning models to predict bias
in different texts. As we continue to use machine learning and text classification models in day to day life, it is important to understand biases in human texts 
as well as possible machine generated biases.

2. What is interesting about this problem?
In class we have largely looked at data from daily user constructed sources such as google corpus, twitter and IMDB movie reviews. The text in these data bases can be
very subjective as well as grammatically inaccurate. Looking at political documents and speeches is likely to lead to a vastly different collection of vocabulary as well 
as word context. As the political scenes change around the world, understanding classification for political documents and speech is a necessary tool for machine models
to correctly understand and project political language.

3. What is the novel component of your project?
We are going to look at compare the ability of a BERT classification model to predict and understand biases towards certain political ideologies and/or parties using political 
documents and speeches from a couple of different countries that have different political diasporas. Our model will be a BERT model that is specified and trained on hand annotated
political documents to determine if the model can detect certain political biases. Hopefully comparing different political diasporas will give insight into types of political speech
and how it can vary between different systems in hopes of developing better text classification and protection models.

## Data

We will be using the following datasets for our project:

There is a large set of data from different countries and political structures that is compiled and available on a git hub repository by author, Erik Gahner.
[Github link](https://github.com/erikgahner/PolData?tab=readme-ov-file)  
We will likely look at data from American political texts as well as European and Central American countries as those are easier to define political parties.

## Evaluation Strategy

*Need more info here  
We will largely be using an accuracy of log probabilities model here as we have previously with our text classification models. Here we can also use precision and recall. The prediction
scores of our model will be computed against the hand annotations.

## Tentative Week-by-Week Timeline

Week 1 (April 8th - April 14th):
Project Proposal.
Complete data collection and finalize countries of interest.  
Week 2 (April 15th - April 21st):
Complete data preprocessing.
Begin annotations.
Begin to implement base BERT model.  
Week 3 (April 22nd - April 28th):
Refine baseline model.
Complete some annotations.
Test baseline model on small test set of annotations.  
Week 4 (April 29th - May 5th):
Finish annotations.
Train model.  
Week 5 (May 6th - May 12th):
Complete model.
Train again if necessary.
Begin write up/results.  
Week 6 (May 13th - May 19th):
Finish results.
Submit project.

## Intended Deliverable for Project Check-in 2

For project check-in 2, we plan to deliver:  
Implementation of baseline model.
Collection of annotations.
Hypothesis and ideas on improvement of model.