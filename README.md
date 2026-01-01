# Customer Intelligence System  
###Customer Churn Prediction & Customer Segmentation 

## Project Overview
Customer churn is a major challenge for subscription-based businesses such as telecom companies.  
This project builds an **end-to-end Customer Intelligence System** that:

- Predicts **which customers are likely to churn** (Supervised Learning)
- Segments customers into **meaningful behavioral groups** (Unsupervised Learning)
- Combines both results to generate **actionable business retention strategies**

The goal is not only to build models, but to **support real business decision-making**.

---

## Business Objectives
- Identify customers with **high churn risk**
- Understand **different types of customers** based on behavior
- Determine **which segments generate high revenue**
- Design **segment-specific retention strategies**

---

## Dataset
- **Dataset Name:** Telco Customer Churn Dataset  
- **Domain:** Telecommunications  
- **Records:** 7,000+ customers  
- **Features include:**
  - Customer demographics
  - Service usage
  - Contract and payment details
  - Billing information
  - Churn label (Yes / No)

---

## Technologies & Tools
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## Project Approach

### Data Preprocessing & EDA
- Handled missing values and incorrect data types
- Performed Exploratory Data Analysis to understand customer behavior
- Analyzed relationships between tenure, charges, and churn

---

### Churn Prediction (Supervised Learning)
- Built a preprocessing pipeline using:
  - Numerical scaling
  - Categorical encoding
- Trained a **Logistic Regression** model
- Evaluated churn patterns to identify high-risk customers

**Outcome:**  
Predicted customer churn behavior and identified churn drivers.

---

### Customer Segmentation (Unsupervised Learning)
- Removed churn label for unbiased clustering
- Applied **K-Means clustering** on customer behavior data
- Segmented customers into **5 distinct clusters**

**Outcome:**  
Each customer was assigned to a behavioral segment.

---

###  Combining Churn & Segmentation (Business Intelligence)
- Calculated **churn rate per customer segment**
- Calculated **average revenue per segment**
- Identified:
  - High-risk segments
  - High-value segments
  - Stable low-risk segments

---

##  Key Insights

| Segment | Churn Rate | Revenue Level | Interpretation |
|-------|----------|---------------|---------------|
| Cluster 1 | ~47% | Low | Very high churn, early exit customers |
| Cluster 4 | ~50% | Medium | Highest churn risk, urgent attention |
| Cluster 2 | ~16% | Very High | High-value loyal customers |
| Cluster 0 | ~7% | Medium | Stable long-term customers |
| Cluster 3 | ~7% | Low | Low-cost, stable customers |

---

##  Business Recommendations

- **High Churn Segments (Clusters 1 & 4)**
  - Targeted discounts
  - Contract upgrades
  - Personalized retention campaigns

- **High-Value Segment (Cluster 2)**
  - Loyalty rewards
  - Premium support
  - Exclusive service plans

- **Stable Segments (Clusters 0 & 3)**
  - Maintain service quality
  - Upselling and cross-selling opportunities

---

##  Business Impact
This system enables businesses to:
- Reduce customer churn
- Allocate retention budgets effectively
- Protect high-value customers
- Increase overall customer lifetime value

---

##  Conclusion
By combining **supervised learning**, **unsupervised learning**, and **business analysis**, this project demonstrates how machine learning can be applied to solve real-world customer retention problems in a practical and explainable way.

---

## Project Structure
Customer-Intelligence-System/
│── README.md
├── customer_intelligence_system.py
└── requirements.txt
