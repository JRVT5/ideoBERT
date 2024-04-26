## Future Steps
Try to remove trump from sentences and see how the model does
-> could replace with generic names or just take out the names in general

preprocess scripts to take out shortest sentences

add neutral class

pretrain the model on some other similar datasets before finetuning

add creation of precision, recall, and f1 score to the final

try training on different data such as the BigNews with slightly different social media context

add hugging face processing

## Lit review

https://news.illuminating.ischool.syr.edu/2020/11/24/polibert-classifying-political-social-media-messages-with-bert/

Traditionally, supervised ML algorithms have been used to classify political content on social media, adapting NLP models to accommodate the unique features of platforms like Twitter and Facebook. SVM has been particularly effective, achieving F1 scores ranging from 65% to 80%. However, challenges such as the need for large training corpora and the inability to capture contextual nuances persist.

In contrast, BERT, a deep learning model, has demonstrated state-of-the-art performance in various NLP tasks. Despite being trained on Wikipedia and English literature, BERT's bidirectional nature and pre-training objectives make it adaptable to different domains. The paper posits that BERT could enhance classification accuracy even in the context of short, informal social media texts.

The study collected data from Twitter and Facebook posts of US presidential candidates during the 2016 election campaign. Through human annotation and machine classification, the authors evaluated both SVM and BERT models. While SVM showed decent performance, BERT significantly improved classification results, particularly in categories like Attack, Campaigning Information, and Ceremonial.

Furthermore, the study tested the 2016 BERT model on 2020 US presidential candidate tweets, achieving promising results despite changes in the Campaigning Information category. Overall, BERT's application led to a 9.0% improvement in Twitter classification and a 5.7% improvement in Facebook classification, compared to SVM.

The paper concludes by emphasizing the potential of deep learning techniques like BERT in enhancing understanding of human communication on social media. It suggests avenues for future research, including applying BERT to classify responses to candidates' social media content and exploring its effectiveness in political campaigns outside the United States.

https://arxiv.org/pdf/2205.00619.pdf

https://downloads.hindawi.com/archive/2022/3498123.pdf

### Lit Review Notes
Discuss the usage of different SOA models and submodels of BERT -> our use of distilBERT for speed