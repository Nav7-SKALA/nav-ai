from pymongo import MongoClient, DESCENDING
import os

def get_session_data(session_id:str):
    client = MongoClient(os.getenv("MONGOURL"))
    db = client[os.getenv("MONGODB")]
    collection = db[os.getenv("MONGOCOLLECTION")]
    doc = collection.find_one(
        {"sessionId": session_id},
        sort=[("createdAt", DESCENDING)]
    )
    return doc.get("answer") if doc else "None"
    # return doc.get("chat_summary") if doc else None