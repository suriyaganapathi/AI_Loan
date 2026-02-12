"""
MongoDB connection and utilities for data ingestion
"""
import os
from pymongo import MongoClient
import certifi
from pymongo.errors import ConnectionFailure
import logging

logger = logging.getLogger(__name__)

# MongoDB client singleton
_mongo_client = None
_database = None

def get_mongo_client():
    """Get or create MongoDB client"""
    global _mongo_client
    
    if _mongo_client is None:
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MONGO_URI not found in environment variables")
            
            _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connection established successfully")
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating MongoDB client: {e}")
            raise
    
    return _mongo_client

def get_database(db_name="ai_finance_db"):
    """Get database instance"""
    global _database
    
    if _database is None:
        client = get_mongo_client()
        _database = client[db_name]
        logger.info(f"✅ Connected to database: {db_name}")
    
    return _database

def get_call_data_collection():
    """Get the call_data collection"""
    db = get_database()
    return db["call_data"]

def close_mongo_connection():
    """Close MongoDB connection"""
    global _mongo_client, _database
    
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        _database = None
        logger.info("MongoDB connection closed")
