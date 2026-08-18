import sys
print(f"Python: {sys.version}")
try:
    import sqlalchemy
    print(f"SQLAlchemy file: {sqlalchemy.__file__}")
    print(f"SQLAlchemy version: {sqlalchemy.__version__}")
    from sqlalchemy import create_engine
    print("Import create_engine: SUCCESS")
except Exception as e:
    print(f"Error: {e}")
