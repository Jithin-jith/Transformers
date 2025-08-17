# Import the pipeline class from Hugging Face Transformers
from transformers import pipeline  

# Create a text classification pipeline.
# - task="text-classification": specifies we want a sentiment analysis model
# - device="cuda": runs the model on GPU if available (for faster inference)
#   By default, this will load the pretrained model:
#   "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline(task="text-classification",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda")  

# Input text to be classified (positive/negative sentiment)
text = "It is such a great day"

# Run the classifier on the input text and print only the label ("POSITIVE"/"NEGATIVE")
print(classifier(text)[0]['label'])  

# Print the confidence score (probability value for the prediction)
print(classifier(text)[0]['score'])  
