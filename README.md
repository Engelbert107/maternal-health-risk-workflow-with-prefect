# Maternal Health Risk Workflow With Prefect

We aimed to predict health risks by implementing and evaluating several machine learning models. Among them, **K-Nearest Neighbors (KNN)** stood out as the top performer, achieving an impressive **89% accuracy** and **0.93 AUC** on the test set. Among the features analyzed, Age was identified as the most influential factor in predicting the risk of sickness.

The most influential feature in the model was **Age**, which had a significant impact on predicting the risk of sickness.

We have successfully deployed our best-performing K-Nearest Neighbors model using Docker and Flask, while orchestrating the entire workflow with Prefect to ensure an efficient, automated pipeline for future predictions and model updates.



## Key Points and Recommendations for Practical Applications

1. **Focus on Age:**
    - **Younger mothers** should be educated on the importance of prenatal care.
    - **Older mothers** require more frequent check-ups, particularly for conditions like hypertension and gestational diabetes.

2. **Monitor Key Health Indicators:**
    - Regular monitoring of **blood pressure, body temperature**, and **heart rate** is essential.
    - Elevated values should prompt immediate investigation to identify potential complications early.

3. **Promote Healthy Blood Pressure and Sugar Levels**:
    - Encourage mothers to maintain healthy **diastolic blood pressure** and **blood sugar** levels through a balanced diet, regular exercise, and routine medical check-ups.
    - This proactive approach can help prevent conditions like gestational hypertension and diabetes.



## Model Comparison on the Validation set

![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/compare-models.png)
![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/stack-vot.png)


## Get an Overview of how our run is Performing on the Prefect Cloud

![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/running-view.png)


## To run Workflow 

1. Install [Docker](https://docs.docker.com/get-started/get-docker/) on your system
2. Clone this repository
    ```bash
    git clone https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect.git
    ```
3. Install packages
    ```bash
    pip install -r requirements.txt
    ```
4. Run the workflow
    ```bash
    python3 orchestrate.py
    ```
5. Build the Docker custom image from the Dockerfile
    ```bash
    docker build -t my-docker-image-api .
    ```
6. Run the container
    ```bash
    docker run --name my-docker-api -p 5000:5000 my-docker-image-api
    ```
7. Open your browser here :
    ```bash
    http://localhost:5000/apidocs
    ```

## See how it Works With a Demo

![](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/blob/main/images/demo.gif)


## Access this Repository Through the Following Links:

- Access to the [data here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/data)
- Access to the [notebook here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/notebook)
- Access to different [images here](https://github.com/Engelbert107/maternal-health-risk-workflow-with-prefect/tree/main/images)
