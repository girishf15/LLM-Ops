#!/usr/bin/env python3
"""
PostgreSQL setup script for RAG application with pgvector extension.
This script will create the database and enable the pgvector extension.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_embeddings")

def setup_database():
    """Create database and enable pgvector extension."""
    
    # First connect to default postgres database to create our target database
    try:
        print(f"🔗 Connecting to PostgreSQL server at {POSTGRES_HOST}:{POSTGRES_PORT}")
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database="postgres"  # Default database
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        print(f"📊 Creating database: {POSTGRES_DB}")
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{POSTGRES_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {POSTGRES_DB}")
            print(f"✅ Database '{POSTGRES_DB}' created successfully")
        else:
            print(f"📋 Database '{POSTGRES_DB}' already exists")
        
        cursor.close()
        conn.close()
        
        # Now connect to our target database to enable pgvector
        print(f"🔗 Connecting to database: {POSTGRES_DB}")
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Enable pgvector extension
        print("🧩 Enabling pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✅ pgvector extension enabled successfully")
        
        # Verify extension is installed
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()
        if result:
            print("🎯 pgvector extension is active and ready to use")
        else:
            print("⚠️  Warning: pgvector extension not found")
        
        cursor.close()
        conn.close()
        
        print("🎉 PostgreSQL setup completed successfully!")
        print(f"💡 Connection string: postgresql://{POSTGRES_USER}:***@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Setting up PostgreSQL database for RAG application...")
    print("=" * 60)
    
    if setup_database():
        print("\n🎯 Next steps:")
        print("1. Install Python packages: pip install -r requirements.txt")
        print("2. Run pdf_processor.py to load documents into the database")
        print("3. Start the Flask app: python app.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
