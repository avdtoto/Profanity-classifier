# Profanity Detection with BERT-base model

In this project, I used RuBERT to build a reliable classifier for categorizing text into two classes: normal text and profane text. The project involved training a sequence classification model, hyperparameter optimization, and experiments with various preprocessing methods (augmentation with Russian comments from https://kaspi.kz and artificial insertion of profane words, using the larger ruBERT-large model, and more complex preprocessing). The best results, in my case, were achieved with the base model without additional data augmentation or complex preprocessing, which is the version presented here.

## Key Features

- Pretrained model: used `DeepPavlov/rubert-base-cased` for sequence classification
- Tokenization: input text tokenized with AutoTokenizer
- Hyperparameter optimization: used `Optuna` to tune learning rate, number of steps, and epochs
- Handling class imbalance: applied weighted loss function to mitigate dataset imbalance
- Model evaluation: F1 score

## Model Training

- Optimizer: AdamW with learning rate `6.6875e-06`
- Scheduler: adjusted learning rate for training stability (`num warmup steps: 357`)
- Training epochs: `3`

## Results

The best version of the model, without additional augmentation or preprocessing, achieved the following results:

- Test F1 score: 0.980
- Original Kaggle test F1 score: 0.9542

## Files

- [wb_winter_school_best_version.ipynb](/wb_winter_school_best_version.ipynb) notebook with results 
- [submission.csv](/submission.csv) predictions on the original test set
- [wb_winter_school.pdf](/wb_winter_school.pdf) presentation of the work
- [kaspi](/kaspi_short.csv) dataset used for augmentation

