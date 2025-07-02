from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from covert_to_num import X, y

#Split data into training & testing sets (80% training / 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Logistic Regression Model
model = LogisticRegression()

#Train The Model Using The Training Data
def train_model():
    model.fit(X_train, y_train)

    print('\nModel Training Completed!')