"""
Coding Agent Tools
Provides basic capabilities for the agent to interact with the environment.
"""

import os

class ToolResult:
    def __init__(self, content: str, success: bool = True):
        self.content = content
        self.success = success

class FileTool:
    @staticmethod
    def read_file(path: str) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(f"Error: File {path} does not exist.", False)
            with open(path, 'r', encoding='utf-8') as f:
                return ToolResult(f.read())
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", False)

    @staticmethod
    def write_file(path: str, content: str) -> ToolResult:
        try:
            # Basic security: prevent writing outside of current directory tree in a real app
            # For this demo, we allow it but create dirs if needed.
            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
                
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(f"Successfully wrote to {path}")
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", False)

class LinterTool:
    @staticmethod
    def check_syntax(code: str) -> ToolResult:
        try:
            import ast
            ast.parse(code)
            return ToolResult("Syntax OK")
        except SyntaxError as e:
            return ToolResult(f"Syntax Error: {e}", False)
        except Exception as e:
            return ToolResult(f"Linter Error: {e}", False)
