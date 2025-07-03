from pymongo import MongoClient, DESCENDING
import os
from bson import ObjectId 
from datetime import datetime

def get_session_data(session_id: str):
    """MongoDB에서 세션 데이터 가져오기"""
    try:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB")]
        collection = db[os.getenv("MONGO_COLLECTION")]
        doc = collection.find_one(
            {"sessionId": session_id},
            sort=[("createdAt", DESCENDING)]
        )
        client.close()
        return doc if doc else None
    except Exception as e:
        print(f"MongoDB 조회 오류: {e}")
        return None

def get_latest_chat_summary(session_id: str):
    """해당 세션의 최신 chat_summary 가져오기"""
    try:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB")]
        collection = db[os.getenv("MONGO_COLLECTION")]
        doc = collection.find_one(
            {"sessionId": session_id},
            sort=[("createdAt", DESCENDING)]
        )
        client.close()
        return doc.get("chat_summary", "") if doc else ""
    except Exception as e:
        print(f"MongoDB chat_summary 조회 오류: {e}")
        return ""

def get_rolemodel_data(rolemodel_id: str):
    """MongoDB에서 롤모델 데이터 가져오기"""
    try:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB")]
        collection = db["role_model"]  # rolemodels -> role_model로 변경
        doc = collection.find_one({"_id": ObjectId(rolemodel_id)})  # ObjectId 추가
        client.close()
        return doc if doc else None
    except Exception as e:
        print(f"롤모델 데이터 조회 오류: {e}")
        return None

def save_session_data(session_id: str, user_id: str, rolemodel_id: str, user_input: str, ai_response: str, chat_summary: str = ""):
    """MongoDB에 세션 데이터 저장"""
    try:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB")]
        collection = db[os.getenv("MONGO_COLLECTION")]
        
        doc = {
            "sessionId": session_id,
            "userId": user_id if user_id else "",  # user_id가 없어도 저장
            "rolemodelId": rolemodel_id if rolemodel_id else "",
            "userInput": user_input,
            "answer": ai_response,
            "chat_summary": chat_summary,  # chat_summary 추가
            "createdAt": datetime.now()
        }
        
        collection.insert_one(doc)
        client.close()
        return True
    except Exception as e:
        print(f"MongoDB 저장 오류: {e}")
        return False

def update_session_user_id(session_id: str, user_id: str):
    """기존 세션에 user_id 업데이트"""
    try:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB")]
        collection = db[os.getenv("MONGO_COLLECTION")]
        
        collection.update_many(
            {"sessionId": session_id},
            {"$set": {"userId": user_id}}
        )
        client.close()
        return True
    except Exception as e:
        print(f"MongoDB 업데이트 오류: {e}")
        return False

