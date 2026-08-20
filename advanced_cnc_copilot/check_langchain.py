import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.path}")
try:
    import langchain
    print(f"LangChain Version: {langchain.__version__}")
    print(f"LangChain File: {langchain.__file__}")
    import langchain.llms
    print("langchain.llms: FOUND")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
