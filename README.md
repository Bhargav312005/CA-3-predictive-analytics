# E-Commerce Product Return Prediction Using Machine Learning

## CA3 Predictive Analytics Project

**Student:** Bhargav Sor  
**Institute:** Symbiosis Institute of Technology, Nagpur  
**Course:** Predictive Analytics

---

## 1. Project Overview

This project develops an end-to-end machine learning model to predict whether an e-commerce order will be returned.

The project follows the KDD / predictive modelling workflow:

1. Problem identification
2. Data understanding
3. Data preprocessing
4. Feature engineering
5. Exploratory data analysis
6. Model development
7. Model evaluation
8. Interpretation
9. Prediction and risk ranking

The project also identifies factors associated with product returns and evaluates whether the available variables provide useful predictive signal.

## 2. Problem Statement

To develop an end-to-end predictive model that predicts whether an e-commerce product will be returned by a customer based on customer characteristics, product attributes, purchase history, browsing behaviour, and delivery-related factors.

### Objectives

- Clean and preprocess the e-commerce dataset.
- Identify factors associated with product returns.
- Engineer useful predictive features.
- Compare multiple machine learning models.
- Evaluate models using classification and ranking metrics.
- Interpret the best-performing model.
- Demonstrate how return-risk predictions can support business decisions.

## 3. Dataset

**Source:** Kaggle E-Commerce Dataset  
https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset

The project uses:

- `train.csv` — 200,000 rows, 16 columns
- `test.csv` — 50,000 rows, 15 columns
- `sample_submission.csv` — 50,000 rows, 2 columns

### Target

`returned`

- `0` = Not Returned
- `1` = Returned

The training data has an overall return rate of approximately **47.46%**.

## 4. Features

The dataset includes:

- `customer_age`
- `product_price`
- `discount_percent`
- `product_rating`
- `past_purchase_count`
- `past_return_rate`
- `delivery_delay_days`
- `session_length_minutes`
- `num_product_views`
- `device_type`
- `product_category`
- `shipping_method`
- `payment_method`
- `used_coupon`

`order_id` is treated as an identifier and excluded from predictive modelling.

## 5. Data Preprocessing

The pipeline includes:

- Detection of invalid numerical values.
- Conversion of impossible values to missing values.
- Median imputation.
- Missing-value indicators.
- Scaling of numerical variables where required.
- One-hot encoding of categorical variables.
- Stratified train/validation splitting.
- Identifier leakage prevention.

Validation rules are applied to variables such as product price, discount percentage, product rating, past return rate, session length, and product views.

## 6. Feature Engineering

The project creates:

- `final_price`
- `discount_amount`
- `expected_prior_returns`
- `views_per_minute`
- `is_late` in the post-delivery scenario

Two feature-availability scenarios are examined:

### Post-delivery scenario
Includes `delivery_delay_days`. This is the primary scenario used for the main project results.

### Checkout-time scenario
Excludes `delivery_delay_days`, since delivery delay is only known after delivery.

## 7. Machine Learning Models

The following models are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. HistGradientBoosting
5. XGBoost

Hyperparameter tuning uses cross-validation with ROC-AUC as the primary tuning metric.

## 8. Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Lift by risk decile

Risk ranking is also evaluated because the business use case can benefit from identifying high-risk orders rather than relying only on a yes/no classification.

## 9. Results

The best-performing model is **XGBoost**.

### Holdout performance

| Metric | XGBoost |
|---|---:|
| Accuracy | ~57.16% |
| Precision | ~56.24% |
| Recall | ~43.86% |
| F1-score | ~0.493 |
| ROC-AUC | ~0.596 |

The predictive signal is moderate, so the results are reported without artificially increasing performance.

### Risk-ranking result

The highest-risk 10% of orders have an observed return rate of approximately **63.05%**, compared with an overall return rate of approximately **47.46%**.

This gives approximately **1.33× lift** over the overall return rate.

Therefore, the model is more useful as a **return-risk ranking tool** than as a perfect individual yes/no predictor.

## 10. Model Interpretation

Interpretability is performed using:

- Permutation importance
- SHAP analysis
- Feature-level analysis
- Logistic-model odds ratios

Important predictive signals include shipping method, product category, device type, coupon usage, payment method, past purchase count, and delivery delay in the post-delivery scenario.

## 11. Project Structure

```text
CA3_ECommerce_Return_Prediction/
│
├── ECommerce_Return_Prediction_CA3.ipynb
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── results/
│   ├── submission.csv
│   ├── test_probabilities.csv
│   └── figures/
├── README.md
├── requirements.txt
└── CA3 Report File.pdf
```

The notebook can also generate:

```text
submission.csv
test_probabilities.csv
```

## 12. Running the Project

### Google Colab

1. Upload the project ZIP or project files to Colab.
2. Keep the three CSV files inside the `data` folder.
3. Open `ECommerce_Return_Prediction_CA3.ipynb`.
4. Run the notebook from top to bottom.
5. Install the dependencies from `requirements.txt` if required.

The notebook is self-contained; all project code is contained in the notebook.

### Local Jupyter / VS Code

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
ECommerce_Return_Prediction_CA3.ipynb
```

and run the cells sequentially.

## 13. Reproducibility

A fixed random seed is used:

```python
RANDOM_STATE = 42
```

This supports reproducible train/validation splits and experiments.

## 14. Limitations

- The available variables provide only moderate predictive signal.
- Some variables depend on when the prediction is intended to be made.
- `delivery_delay_days` is only available after delivery and should not be used for a genuine checkout-time prediction.
- The model should therefore not be interpreted as a highly accurate individual-return predictor.
- The project focuses on predictive modelling rather than live production deployment.

## 15. Future Scope

Possible improvements include:

- Richer customer behavioural history.
- Product-level return history.
- Seller-level information.
- Customer review/text features.
- Real-time browsing behaviour.
- Probability calibration.
- Additional ensemble methods.
- API or dashboard deployment.
- Validation on a new real-world e-commerce dataset.

## 16. Conclusion

This project demonstrates a complete predictive analytics workflow for e-commerce product return prediction.

Multiple machine learning models were developed and evaluated using a consistent preprocessing and validation framework. XGBoost provided the strongest overall performance, achieving a holdout ROC-AUC of approximately 0.596.

Although predictive accuracy is moderate, the model demonstrates useful ranking capability: the highest-risk 10% of orders have a substantially higher return rate than the overall dataset. This suggests that predictive analytics can help prioritise high-risk orders and support targeted business actions.

## 17. Academic Deliverables

- Jupyter Notebook
- Source code
- Dataset files
- Requirements file
- README
- CA3 report
- Generated prediction outputs
