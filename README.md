Here’s your README rewritten in **GitHub-friendly Markdown** — clean formatting, emoji headers for readability, fenced code blocks, and compatible with GitHub’s Markdown renderer:

---

# 🧠 Machine Learning Modelling of Utility and Probability Weighting in Decision Theory

This repository contains the **reproduction package** for the working paper *"Machine Learning Modelling of Utility and Probability Weighting in Decision Theory"*.

The project investigates how machine learning methods can model and predict human decision-making under risk, comparing traditional economic frameworks (like **Cumulative Prospect Theory**) with **data-driven** and **hybrid** approaches.

---

## 📘 Overview

The goal is to compare three broad classes of models:

1. **Economic Models** – grounded in behavioral economics (e.g., Cumulative Prospect Theory).
2. **Pure Machine Learning Models** – fully data-driven (e.g., random forests, regularized regressions).
3. **Hybrid Models** – neural networks that embed economic theory into their structure or loss function.

This comparison provides insights into how well each class captures **utility curvature** and **probability weighting**, two central features of decision-theoretic behavior.

---

## 📁 Repository Structure

| Script                             | Description                                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DS_ML_Data.py`                    | Prepares and preprocesses the dataset for training and evaluation. Includes feature construction and transformations related to decision-theoretic variables. |
| `DS_ML_Data_2.py`                  | Alternate or extended data preparation script; may include additional cleaning or feature engineering routines.                                               |
| `DT-ML MLE.py`                     | Implements Maximum Likelihood Estimation (MLE) for parameter estimation in decision-theoretic models.                                                         |
| `DT-ML NN.py`                      | Defines and trains hybrid neural network models incorporating economic theory (e.g., constrained layers or custom loss terms).                                |
| `DT-ML Regularized Regressions.py` | Trains regularized regression models (LASSO, Ridge, Elastic Net) as baseline comparators.                                                                     |
| `DT-ML Trees.py`                   | Implements tree-based ML models such as Random Forests and Gradient Boosted Trees.                                                                            |

---

## ⚙️ Requirements

This project is written in **Python 3.x**.
Core dependencies include:

```bash
numpy
pandas
scikit-learn
tensorflow  # or pytorch
matplotlib
seaborn
scipy
```

To install dependencies:

```bash
pip install -r requirements.txt
```

*(If not present, create `requirements.txt` listing the above packages.)*

---

## 🚀 Usage

To reproduce results from the paper:

1. **Prepare the data**

   ```bash
   python DS_ML_Data.py
   # or
   python DS_ML_Data_2.py
   ```

2. **Train and evaluate models**

   ```bash
   python DT-ML Trees.py
   python DT-ML Regularized Regressions.py
   python DT-ML NN.py
   ```

3. **Analyze outputs**

   Model metrics, parameter estimates, and figures are saved in the designated results folders or printed to the console.

---

## 📊 Outputs

The scripts generate:

* Comparative performance metrics (accuracy, RMSE, log-likelihood)
* Visualizations of **utility** and **probability weighting functions**
* Tables comparing **economic**, **ML**, and **hybrid** models

---

## 🧩 Research Context

This work contributes to bridging **behavioral economics** and **machine learning**, by:

* Testing empirical validity of decision-theoretic assumptions
* Using ML to discover flexible, nonparametric patterns in choice behavior
* Building interpretable hybrid models combining structure and prediction power

---

## 📄 Citation

If you use this code or reproduce the results, please cite:

> [Author Name(s)]. *Machine Learning Modelling of Utility and Probability Weighting in Decision Theory.* Working Paper, [Institution], [Year].

---

## 🪪 License

This project is released under the [MIT License](LICENSE).

---

Would you like me to **add shields and badges** (e.g., Python version, license, last update, citation DOI) at the top — commonly used for GitHub repos? They make it look more professional and academic-ready.
