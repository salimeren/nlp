# nlp
Author : Salim Eren Yılmaz

ParliamentaryClassification

Overview

This project provides a Python implementation for classifying parliamentary speeches based on their political orientation and power dynamics (e.g., government or opposition). The code leverages the XLM-RoBERTa model for sequence classification tasks.

Features

Political Orientation Classification: Classify texts as left-wing or right-wing.
Power Classification: Identify whether a speech is from the government or the opposition.
Class Balancing: Uses oversampling to handle imbalanced datasets.
Custom Training: Implements class-weighted loss for improved handling of imbalanced labels.
Device Compatibility: Automatically utilizes GPU if available.

Dataset

Dataset can be found at https://zenodo.org/records/10450641

