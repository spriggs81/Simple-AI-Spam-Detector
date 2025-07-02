from covert_to_num import vectorizer
from train_the_ai import train_model, model

train_model()

#Predict on new, unseen data
new_emails = [
    "You have won a lottery ticket!",
    "Hi team, meeting at 3 PM.",
    "Get your FREE money now!",
    "Confirming your appointment details.",
    "LIMITED TIME OFFER! Click here.",
]

#Transform new emails using vectorizer fitted on training data
new_transformed_emails = vectorizer.transform(new_emails)

#Make predictons
new_predictions_numeric = model.predict(new_transformed_emails)

#Convert Numerical Predictions Back To Original Labels for Readability
reverse_label_mapping = {0: "not spam", 1: "spam"}
new_predictions_text = [reverse_label_mapping[pred] for pred in new_predictions_numeric]

print("\n*****  Predictions on New Emails  *****")
for i, email in enumerate(new_emails):
    print(f"'{email}'  -->  Prediction: {new_predictions_text[i]}")