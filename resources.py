import pandas as pd
from connectDatabase import MongoConnect
from config import USERNAME, PASSWORD, DB, COLLECTION
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

Generic_Questions = pd.read_pickle('GenericQuestions.pkl')

print("Loading models...")
Question_Generator = pipeline('text-generation', model='Question-Generation')
Zeroshot_Classifier = pipeline(
    "zero-shot-classification", model="Zero_Shot-Classification")
Sentiment_Tokenizer = AutoTokenizer.from_pretrained(
    "Sentiment-Classification")
Sentiment_Model = AutoModelForSequenceClassification.from_pretrained(
    "Sentiment-Classification")
Sentiment_Analysis = (Sentiment_Tokenizer, Sentiment_Model)
print("Models loaded successfully!")

print("Connecting with database...")
User_Database = MongoConnect(USERNAME, PASSWORD, DB, COLLECTION)
print("Database connected successfully!")
