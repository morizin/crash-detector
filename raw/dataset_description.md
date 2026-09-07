# Dataset

The dataset is derived from the Kaggle Success Factors Dataset 100k.
link: https://www.kaggle.com/datasets/likithagedipudi/kaggle-success-factors-dataset-100k/data


## Overview

The dataset itself contains 62733 rows of notebook related records, containing notebook metadata like code lines, markdown ratio, visualization count, libraries used, etc, while also some creater related information like author tier, author followers, author notebooks count. The target variable is the upvote count of the notebook (upvotes). 

Note that to adapt the dataset into an upvote prediction problem, we have made some modifications to the original dataset.

The dataset was originally derived from a raw dataset containing both the dataset and notebook related record. We opted out on dataset related records for this competition to simplify the problem. People often upvote for notebook and dataset for completely different reasons, so this dataset in its original form represent two distinct prediction problems. Dataset itself is often harder to create and its upvote count is often not directly tied to the quality of the dataset, but more on the usefulness of the dataset for certain tasks, which means that the features we have may not be sufficient to predict the upvote count for dataset related records. Thus we will only focus on notebook related records.

Additionally, we have dropped a few features that could have potentially leak the target variable. For example, features like "number of comments", "number of forks" are likely correlated with the upvote count and would ofen appear after the creation of the notebook which introduces temporal leakage issues, and thus we have removed them from the dataset to avoid data leakage.

## File Structure

- **train.csv** - train set (with labels)
- **test.csv** - test set (unlabeled)


## Features

FEATURES:
- **content_id** - an anonymous id unique to a notebook
- **title** - the title of the notebook
- **author_username** - the username of the author of the notebook
- **author_tier** - the tier of the author (Novice, Contributor, Master, Grandmaster) from Kaggle
- **author_followers** - the number of followers of the author on Kaggle (currently)
- **author_notebooks_count** - the number of notebooks the author has published
- **author_datasets_count** - the number of datasets the author has published
- **primary_topic** - the primary topic of the notebook 
- **all_topics** - all topics of the notebook
- **programming_language** - the programming language of the notebook
- **is_competition_related** - whether the notebook is related to a competition
- **created_date** - the date the notebook was created (dd/mm/yyyy)
- **last_updated** - the date the notebook was last updated (dd/mm/yyyy)
- **days_since_creation** - the number of days since the notebook was created
- **update_count** - the number of times the notebook was updated
- **execution_time_seconds** - the execution time of the notebook in seconds
- **code_lines** - the number of code lines in the notebook
- **markdown_ratio** - the ratio of markdown cells in the notebook
- **visualization_count** - the number of visualizations in the notebook
- **libraries_used** - the libraries used in the notebook
- **uses_gpu** - whether the notebook uses GPU
- **file_size_mb** - the file size of the notebook in MB
TARGET:
- **upvotes (TARGET)** - the number of upvotes the notebook has received


## Data Characteristics
- Missing values: Lot of them in features like file_format, column_count, row_count, license_type, etc.
- Target distribution: The target distribution is right skewed, with a lot of notebooks having upvotes below 100.
- The dataset contains numerical, categorical, date, and text features. This really varied feature set is designed to challenge the participants to use the features effectively and creatively.
- Text Feature: There is a pure text "title" column, which can be utilized if embedding model or tfidf like methods are used.
- Concatenated Categorical Features: The all_topics and libraries_used column include multiple categorical values concatenated with "|". 
- Date Features: The created_date and last_updated feature are all in dd/mm/yyyy format. 
