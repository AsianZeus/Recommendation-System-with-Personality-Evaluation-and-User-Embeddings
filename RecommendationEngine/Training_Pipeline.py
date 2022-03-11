from config import config
from Prepare_Training_Data import matches_to_matches_triplet, database2csv, maphex2int
from Training_Embeddings import train_embeddings
from Training_Model import train_model
from pymongo import MongoClient

try:
    client = MongoClient(
        f"mongodb+srv://{config.USERNAME}:{config.PASSWORD}@{config.HOST}/{config.DB}?retryWrites=true&w=majority")
    db = client[config.DB]
    collection = db[config.COLLECTION]
    print("Connected to the database!")
except Exception as e:
    print("Unable to connect to the database!", e)
    exit()

print("Fetching data from database...")
print("Mapping hex2int...")
hex2int = maphex2int(collection, config.ids_path)
print("Converting data to csv...")
database2csv(collection, config.train_data_path, hex2int)
print("Preparing training data...")
matches_to_matches_triplet(config.train_data_path, config.d2v_train_data_path)
print("Training embeddings...")
train_embeddings()
print("Training model...")
train_model()
print("Done!")
