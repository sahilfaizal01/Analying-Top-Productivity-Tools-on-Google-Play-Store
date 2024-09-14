# Analysing Top Productivity Tools on Google Play Store
Using python data analysis and predictive modelling using BERT to find sentiments from reviews

# Note:
* Since BERT can only handle up to 512 tokens in one pass, you would need to split the text into multiple chunks, each containing no more than 512 tokens.
* The input IDs are numerical representations of the text that the model can understand. When you tokenize a sentence (e.g., "BERT is great"), each token is mapped to a unique ID from the tokenizer's vocabulary.
* The attention mask is a binary mask (i.e., a list of 1s and 0s) that tells the model which tokens to pay attention to and which to ignore. Specifically:
  * 1 indicates that the corresponding token should be attended to (i.e., it's part of the actual text).
  * 0 indicates that the corresponding token is just padding and should be ignored by the model.
