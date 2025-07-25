## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Accord emphasizes rigorous risk measurement and regulatory compliance. This requires credit risk models to be interpretable and well-documented, ensuring transparency for audits and regulatory reviews. Interpretable models allow financial institutions to justify lending decisions and manage risk exposure effectively.

### 2. Proxy Variable for Credit Risk
Since our dataset lacks a direct "default" label, we must create a proxy variable (e.g., high-risk customers based on engagement metrics). This is necessary to train and evaluate models. However, predictions based on proxies may not fully capture true default risk, introducing potential business risks such as misclassification and regulatory scrutiny.

### 3. Model Trade-offs in Regulated Contexts
Simple models (e.g., Logistic Regression with WoE) offer interpretability and regulatory acceptance but may sacrifice predictive power. Complex models (e.g., Gradient Boosting) can improve accuracy but are harder to interpret and justify. In regulated environments, the trade-off is between transparency and performance, with a preference for models that balance both.