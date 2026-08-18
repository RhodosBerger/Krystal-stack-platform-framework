import sqlalchemy
print(f"Version: {sqlalchemy.__version__}")
try:
    from sqlalchemy import create_engine
    print("create_engine: FOUND")
except ImportError:
    print("create_engine: MISSING")

try:
    from sqlalchemy import create_all
    print("create_all: FOUND")
except ImportError:
    print("create_all: MISSING")
