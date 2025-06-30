from pymongo import MongoClient, DESCENDING
import os

def get_session_data(session_id:str):
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client[os.getenv("MONGO_DB")]
    collection = db[os.getenv("MONGO_COLLECTION")]
    doc = collection.find_one(
        {"sessionId": session_id},
        sort=[("createdAt", DESCENDING)]
    )
    # return doc.get("answer") if doc else "None"
    return doc.get("chat_summary") if doc else None