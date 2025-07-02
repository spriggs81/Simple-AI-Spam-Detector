from sklearn.metrics import accuracy_score, classification_report
from train_the_ai import X_test, y_test, model, train_model

train_model()

#Make predictions on the test info
y_pred = model.predict(X_test)

#Evalulate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

#Print a detail classification report
print("\nclassification Report:\n", classification_report(y_test, y_pred, target_names=["not spam", "spam"]))