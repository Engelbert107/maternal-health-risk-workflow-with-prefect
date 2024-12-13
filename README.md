# Maternal Health Risk Workflow With Prefect

In this project, we implemented and evaluated several machine learning models to predict health risks. Among them, **K-Nearest Neighbors (KNN)** stood out as the top performer, achieving an impressive **89% accuracy** and **0.93 AUC** on the test set. KNN showed the fewest errors, as seen in the confusion matrix, and demonstrated strong predictive capability, effectively distinguishing between high- and low-risk classes. The most influential feature in the model was **Age**, which had a significant impact on predicting the risk of sickness.

We also orchestrated the workflow for the best-performing model, K-Nearest Neighbors, using **Prefect**, ensuring a robust and automated pipeline for future predictions and model deployment.



## Key Points and Recommendations

1. **Focus on Age:**
    - **Younger mothers** should be educated on the importance of prenatal care.
    - **Older mothers** require more frequent check-ups, particularly for conditions like hypertension and gestational diabetes.

2. **Monitor Key Health Indicators:**
    - Regular monitoring of **blood pressure, body temperature**, and **heart rate** is essential.
    - Elevated values should prompt immediate investigation to identify potential complications early.

3. **Promote Healthy Blood Pressure and Sugar Levels**:
    - Encourage mothers to maintain healthy **diastolic blood pressure** and **blood sugar** levels through a balanced diet, regular exercise, and routine medical check-ups.
    - This proactive approach can help prevent conditions like gestational hypertension and diabetes.


## Model comparison 

![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/compare-models.png)


## How to test our workflow ?
1. Download the project repository to explore the code and documentation.
2. Install packages.
    ```bash
    pip install -r requirements.txt
    ```
3. Run the workflow
    ```bash
    python3 orchestrate.py
    ```

### Get an overview of how our run is performing on the cloud

   ![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/running-view.png)


## Access this repository through the following links:
- Access to the [data here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/data)
- Access to the [notebook here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/notebook)
- Access to different [images here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/images)
