# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn (`RandomForestClassifier`, `n_estimators=100`, `random_state=42`). It was developed as part of a FastAPI-based ML pipeline deployment project. The model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment features from the U.S. Census dataset.

## Intended Use

This model is intended for educational and research purposes to demonstrate the deployment of a machine learning pipeline using FastAPI. It should not be used to make real-world decisions about individuals' finances, employment, or creditworthiness. Appropriate use cases include learning about ML model deployment, REST API design, and model performance evaluation across demographic slices.

## Training Data

The model was trained on the UCI Census Income dataset (`census.csv`), which contains approximately 32,000 records extracted from the 1994 U.S. Census database. The data includes 14 features covering demographic and employment attributes such as age, education, occupation, marital status, and native country. An 80/20 train-test split was applied with `random_state=42`. Categorical features were encoded using a `OneHotEncoder` and the binary label (`salary`) was encoded using a `LabelBinarizer`.

## Evaluation Data

The model was evaluated on the held-out 20% test split (~6,513 records), using the same preprocessing pipeline (fitted encoder and label binarizer from training). No separate external validation dataset was used.

## Metrics

Model performance is evaluated using **precision**, **recall**, and **F1 score** (beta=1).

Overall performance on the test set:
- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

Selected performance on categorical slices (see `slice_output.txt` for full results):

| Feature | Slice | Precision | Recall | F1 |
|---|---|---|---|---|
| workclass | Federal-gov | 0.7971 | 0.7857 | 0.7914 |
| workclass | Private | 0.7376 | 0.6404 | 0.6856 |
| workclass | Self-emp-not-inc | 0.7064 | 0.4904 | 0.5789 |
| education | Bachelors | 0.7523 | 0.7289 | 0.7404 |
| education | Masters | 0.8271 | 0.8551 | 0.8409 |
| education | Doctorate | 0.8644 | 0.8947 | 0.8793 |
| education | HS-grad | 0.6594 | 0.4377 | 0.5261 |
| sex | Female | 0.7229 | 0.5150 | 0.6015 |
| sex | Male | 0.7445 | 0.6599 | 0.6997 |
| race | White | 0.7404 | 0.6373 | 0.6850 |
| race | Black | 0.7273 | 0.6154 | 0.6667 |
| race | Asian-Pac-Islander | 0.7857 | 0.7097 | 0.7458 |
| race | Amer-Indian-Eskimo | 0.6250 | 0.5000 | 0.5556 |

Performance is generally stronger for higher-education groups and for groups with larger sample sizes. Several small-count slices (e.g., `native-country` values with fewer than 10 samples) show perfect or near-perfect scores that likely reflect insufficient data rather than genuine model performance.

## Ethical Considerations

The Census Income dataset contains sensitive demographic attributes including race, sex, and national origin. The slice analysis reveals meaningful performance disparities across these groups. For example, recall for females (0.5150) is notably lower than for males (0.6599), indicating the model is less effective at identifying high-income women. Similarly, performance varies across racial groups, with the American Indian/Eskimo slice showing lower recall than other groups. These disparities reflect biases present in the historical data and should be carefully considered before using this model in any context where fairness matters.

## Caveats and Recommendations

The training data is drawn from the 1994 U.S. Census, making it over 30 years old. Income thresholds, occupational distributions, and societal demographics have changed substantially since then, so the model should not be used to draw conclusions about today's population. Many non-U.S. country slices have very small sample sizes (fewer than 20 records), making their metrics unreliable. The model would benefit from hyperparameter tuning, cross-validation, and evaluation on more recent, representative data before any production use.
