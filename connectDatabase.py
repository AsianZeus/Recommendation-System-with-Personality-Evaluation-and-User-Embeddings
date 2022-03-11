from pymongo import MongoClient
from bson.objectid import ObjectId


class MongoConnect:

    def __init__(self, username, password, db, collection, host="localhost"):
        try:
            self.client = MongoClient(
                f"mongodb+srv://{username}:{password}@{host}/{db}?retryWrites=true&w=majority")
            self.db = self.client[db]
            self.collection = self.db[collection]
        except Exception as e:
            print("Unable to connect to the database!", e)

    def get_qna(self, id):
        qna = self.collection.find_one(
            {"_id": ObjectId(id)}, {"_id": 0, "qna": 1})
        if qna is None:
            return None
        return qna['qna']

    def get_interests(self, id):
        interests = self.collection.find_one(
            {"_id": ObjectId(id)}, {"_id": 0, "interests": 1})
        if interests is None:
            return None
        return interests['interests']

    def get_personality_scores(self, id):
        personality_scores = self.collection.find_one(
            {"_id": ObjectId(id)}, {"_id": 0, "personality_scores": 1})
        if personality_scores is None:
            return None
        return personality_scores['personality_scores']

    def verify_user(self, id):
        try:
            ObjectId(id)
        except Exception as e:
            return False
        user = self.collection.find_one({"_id": ObjectId(id)})
        if user is None:
            return False
        return True

    def count_liked_users(self, id):
        liked_users = self.collection.find_one(
            {"_id": ObjectId(id)}, {"_id": 0, "liked_users": 1})
        if liked_users is None:
            return 0
        return len(liked_users['liked_users'])

    def count_disliked_users(self, id):
        disliked_users = self.collection.find_one(
            {"_id": ObjectId(id)}, {"_id": 0, "disliked_users": 1})
        if disliked_users is None:
            return 0
        return len(disliked_users['disliked_users'])
