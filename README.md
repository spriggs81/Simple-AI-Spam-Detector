# Simple AI Spam Detector

## Project Overview

This project demonstrates a basic Artificial Intelligence model designed to classify email titles as either "spam" or "not spam." It serves as a foundational example of a binary text classification problem using traditional machine learning techniques. The project covers essential steps in an AI/ML workflow, including data preparation, feature engineering, model training, evaluation, and making predictions on new, unseen data.

## Features

* **Data Preparation:** Organizes labeled examples of spam and non-spam email titles.
* **Text Preprocessing:** Converts raw text into numerical features using `CountVectorizer`.
* **Model Training:** Employs a Logistic Regression model to learn patterns from the prepared data.
* **Model Evaluation:** Assesses the model's performance on unseen test data, providing accuracy and a detailed classification report.
* **Real-time Prediction:** Demonstrates how to use the trained model to predict the category of new email titles.

## Project Structure

The project is organized into several modular Python files for clarity and better understanding of each step in the AI pipeline.

spamdetectorapp/

├── training_data.py

├── covert_to_num.py

├── train_the_ai.py

├── evaluate_model.py

├── new_data.py

└── requirements.txt

### File Breakdown:

1.  **`training_data.py`**
    * **Purpose:** Stores the raw dataset used for training the AI. This file contains a list of tuples, where each tuple includes an email title and its corresponding label ("spam" or "not spam").
    * **Snippet:**
        ```python
        # Basic Data to Train the AI to understand what's spam or not
        data = [
            # Our Spam Examples
            ("Win a Free prize!", "spam"),
            # Our Not Spam Examples
            ("Hello, how are you?", "not spam"),
            # ... (400+ more entries) ...
        ]
        ```

2.  **`covert_to_num.py`**
    * **Purpose:** Handles the feature engineering process. It converts the raw email text into a numerical format (Count Vectorization) that the machine learning model can understand. It also maps the "spam"/"not spam" labels to numerical values (0s and 1s).
    * **Snippet:**
        ```python
        #Fit the vectorizer to our texts & transform them into numerical features
        X = vectorizer.fit_transform(texts)

        #Convert labels to numerical format (0 for 'not spam' & 1 for 'spam')
        label_mapping = {"not spam": 0, "spam": 1}

        #using pandas for easy mapping
        #'y' will be the target labels (0s & 1s)
        y = pd.Series(labels).map(label_mapping).values

        # ... (test_data function for debugging/display) ...
        ```

3.  **`train_the_ai.py`**
    * **Purpose:** Initializes and trains the Logistic Regression machine learning model. It imports the numerical data (`X` and `y`) and splits it into training and testing sets (80% training, 20% testing) to ensure a fair evaluation.
    * **Snippet:**
        ```python
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from covert_to_num import X, y

        #Split data into training & testing sets (80% training / 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Initialize the Logistic Regression Model
        model = LogisticRegression()

        def train_model():
            model.fit(X_train, y_train)
            print('\nModel Training Completed!')
        ```

4.  **`evaluate_model.py`**
    * **Purpose:** Evaluates the performance of the trained AI model. It makes predictions on the unseen test data (`X_test`) and calculates metrics such as accuracy and provides a detailed classification report to assess how well the model generalizes.
    * **Snippet:**
        ```python
        from sklearn.metrics import accuracy_score, classification_report
        from train_the_ai import X_test, y_test, model, train_model

        train_model() # Ensure model is trained before evaluation

        #Make predictions on the test info
        y_pred = model.predict(X_test)

        #Evalulate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy on Test Set: {accuracy:.2f}")

        #Print a detail classification report
        print("\nclassification Report:\n", classification_report(y_test, y_pred, target_names=["not spam", "spam"]))
        ```

5.  **`new_data.py`**
    * **Purpose:** Demonstrates how to use the trained model to classify new, unseen email titles. It preprocesses the new text using the same `CountVectorizer` fitted during training and outputs the AI's prediction (spam or not spam).
    * **Snippet:**
        ```python
        from covert_to_num import vectorizer
        from train_the_ai import train_model, model

        train_model() # Ensure model is trained before making new predictions

        #Predict on new, unseen data
        new_emails = [
            "You have won a lottery ticket!",
            "Hi team, meeting at 3 PM.",
            # ... (more examples) ...
        ]

        #Transform new emails using vectorizer fitted on training data
        new_transformed_emails = vectorizer.transform(new_emails)

        #Make predictons and print
        # ...
        ```

6.  **`requirements.txt`**
    * **Purpose:** Lists all the Python packages and their exact versions required for this project to run. This ensures reproducibility across different environments.
    * **Snippet (truncated for brevity, actual file contains all 92+ dependencies):**
        ```
        aiohttp==3.11.13
        aiosignal==1.3.2
        numpy==2.2.3
        pandas==2.2.3
        scikit-learn==1.4.2
        # ... (92+ other dependencies) ...
        ```

## Technologies Used

* **Python 3.x**
* **scikit-learn:** For machine learning algorithms (`LogisticRegression`, `CountVectorizer`, `train_test_split`, `accuracy_score`, `classification_report`).
* **NumPy:** (Dependency of scikit-learn/pandas) For numerical operations.
* **Pandas:** For data manipulation (specifically for handling labels).

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/spriggs81/Simple-AI-Spam-Detector.git
    cd Simple-AI-Spam-Detector/spamdetectorapp # Adjust path if your project root is higher
    ```

2.  **Create a virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv venv # or 'python -m venv folder1' if that's what you named it
    ```

3.  **Activate the virtual environment:**
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate # or 'source folder1/bin/activate'
        ```
    * **On Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat # or 'folder1\Scripts\activate.bat'
        ```
    * **On Windows (PowerShell):**
        ```powershell
        venv\Scripts\Activate.ps1 # or 'folder1\Scripts\Activate.ps1'
        ```
    (Your terminal prompt should change to indicate the active environment, e.g., `(venv)`)

4.  **Install dependencies:**
    With your virtual environment activated, install all required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

After setting up the environment and installing dependencies, you can run the different parts of the AI Spam Detector:

* **To run the full evaluation and see new predictions:**
    The `new_data.py` file, as currently structured, calls `train_model()` before making predictions. If you want to see both evaluation and new data predictions in one go, you could create a single `main.py` that imports and calls functions from `evaluate_model.py` and `new_data.py`. For now, you can run them sequentially:

    ```bash
    # Ensure your virtual environment is activated
    python evaluate_model.py
    python new_data.py
    ```

    You can also run `covert_to_num.py`'s `test_data()` function if you uncomment its call or add a `if __name__ == "__main__":` block to it.

## Future Enhancements

* **Larger, More Diverse Datasets:** Utilize publicly available, larger datasets of spam/ham emails for more robust training.
* **Advanced Feature Engineering:** Explore TF-IDF (Term Frequency-Inverse Document Frequency) for more nuanced text representation.
* **Different Classification Algorithms:** Experiment with other `scikit-learn` models like Naive Bayes, Support Vector Machines (SVMs), or Decision Trees.
* **Deep Learning for NLP:** For very large datasets, consider using deep learning frameworks (TensorFlow, PyTorch) with models like Recurrent Neural Networks (RNNs) or Transformers for even more sophisticated text understanding.
* **Hyperparameter Tuning:** Optimize the parameters of the chosen machine learning model for better performance.
* **Integration:** Explore integrating this core logic into a small application (e.g., a web service) to classify emails in real-time.

---
**Note:** This is a simplified educational project. Real-world spam detection systems are far more complex, involving vast datasets, advanced NLP techniques, constant model retraining, and robust infrastructure.
