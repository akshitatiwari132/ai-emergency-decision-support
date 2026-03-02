# AI-Enabled Emergency Decision Support System

Here, I built a decision support system for prioritizing high-severity emergency incidents using environmental signals. The system predicts accident severity using supervised learning and converts model risk outputs into actionable dispatch levels while incorporating responsible AI safeguards. This prototype simulates how AI can support mission-critical decision environments.

My goals are to -
- predict whether an accident is high severity
- prioritize recall for safety-critical outcomes 
- convert predicted risk into operational dispatch levels
- escalate uncertain predictions to human review
- maintain interpretability and responsible AI practices

I used the US Accidents dataset (subset ~200,000 rows) from Kaggle. The features I used were temperature, humidity, pressure, visibility, wind speed, distance, while the target variable was high severity (with a binary classification of severity greater than or equal to 3).

For modeling -
- Logistic Regression (baseline)
- Random Forest (nonlinear model)
- Class imbalance handling (`class_weight="balanced"`)
- Threshold tuning (0.5 to 0.35)

**Results:**
- ROC-AUC: ~0.67  
- High-severity recall (tuned): ~0.70  
- Random Forest outperformed linear baseline  

Tech stack : python • pandas • numpy • scikit-learn • matplotlib
