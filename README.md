# Analysing Top Productivity Tools on Google Play Store
Using python data analysis and predictive modelling using BERT to find sentiments from reviews

# Apps 
![image](https://github.com/user-attachments/assets/2a5fe5fe-5f10-4661-a0ed-9371b52e1f9f)

# Project Description and Working

# Findings

# BERT
What is BERT?
BERT (introduced in this [paper](https://arxiv.org/abs/1810.04805)) stands for Bidirectional Encoder Representations from Transformers. If you don't know what most of that means - you've come to the right place! Let's unpack the main ideas:

Bidirectional - to understand the text you're looking you'll have to look back (at the previous words) and forward (at the next words)
Transformers - The [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper presented the Transformer model. The Transformer reads entire sequences of tokens at once. In a sense, the model is non-directional, while LSTMs read sequentially (left-to-right or right-to-left). The attention mechanism allows for learning contextual relations between words (e.g. his in a sentence refers to Jim).
(Pre-trained) contextualized word embeddings - [The ELMO paper](https://arxiv.org/abs/1802.05365v2) introduced a way to encode words based on their meaning/context. Nails has multiple meanings - fingernails and metal nails.
BERT was trained by masking 15% of the tokens with the goal to guess them. An additional objective was to predict the next sentence. Let's look at examples of these tasks:

## 1) Masked Language Modeling (Masked LM)
The objective of this task is to guess the masked tokens. Let's look at an example, and try to not make it harder than it has to be:

That's [mask] she [mask] -> That's what she said

## 2) Next Sentence Prediction (NSP)
Given a pair of two sentences, the task is to say whether or not the second follows the first (binary classification). Let's continue with the example:

Input = [CLS] That's [mask] she [mask]. [SEP] Hahaha, nice! [SEP]

Label = IsNext

Input = [CLS] That's [mask] she [mask]. [SEP] Dwight, you ignorant [mask]! [SEP]

Label = NotNext

The training corpus was comprised of two entries: Toronto Book Corpus (800M words) and English Wikipedia (2,500M words). While the original Transformer has an encoder (for reading the input) and a decoder (that makes the prediction), BERT uses only the decoder.

BERT is simply a pre-trained stack of Transformer Encoders. How many Encoders? We have two versions - with 12 (BERT base) and 24 (BERT Large).


# Note:
* Since BERT can only handle up to 512 tokens in one pass, you would need to split the text into multiple chunks, each containing no more than 512 tokens.
* The input IDs are numerical representations of the text that the model can understand. When you tokenize a sentence (e.g., "BERT is great"), each token is mapped to a unique ID from the tokenizer's vocabulary.
* The attention mask is a binary mask (i.e., a list of 1s and 0s) that tells the model which tokens to pay attention to and which to ignore. Specifically:
  * 1 indicates that the corresponding token should be attended to (i.e., it's part of the actual text).
  * 0 indicates that the corresponding token is just padding and should be ignored by the model.

# References
1) [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
2) [How to fine-tune BERT model for text classification](https://arxiv.org/pdf/1905.05583)
3) [BERT fine-tuning PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
4) [BERT, ELMo and co.](https://jalammar.github.io/illustrated-bert/)
