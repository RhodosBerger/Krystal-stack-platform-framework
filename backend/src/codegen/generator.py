"""
Code Generation Engine for Dev-conditional Server Engine
Generates FastAPI applications from workflow configurations
"""

import os
import shutil
import json
import ast
import logging
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
import tempfile
import zipfile

from ..config import settings
from ..storage.database import AsyncSessionLocal
from ..storage.models import GeneratedProject

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Main code generation engine"""

    def __init__(self):
        self.templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates")
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    @classmethod
    async def generate_project_async(
        cls,
        project_id: int,
        workflow_data: Dict[str, Any],
        template_name: str = "fastapi_basic",
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """Generate project asynchronously"""
        generator = cls()
        await generator._generate_project(project_id, workflow_data, template_name, custom_config)

    async def _generate_project(
        self,
        project_id: int,
        workflow_data: Dict[str, Any],
        template_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """Internal method to generate project"""
        try:
            # Update project status in database
            await self._update_project_status(project_id, "generating")

            # Extract project configuration from workflow
            project_config = self._extract_project_config(workflow_data, custom_config)

            # Generate project structure
            project_structure = await self._generate_project_structure(
                project_config, template_name
            )

            # Create project directory
            project_dir = os.path.join(settings.GENERATED_PROJECTS_DIR, f"project_{project_id}")
            os.makedirs(project_dir, exist_ok=True)

            # Generate files
            generated_files = await self._generate_files(
                project_dir, project_structure, project_config
            )

            # Create ZIP file
            zip_path = await self._create_project_zip(project_dir, project_id)

            # Update project record in database
            await self._save_project_result(
                project_id, project_structure, zip_path, template_name
            )

            await self._update_project_status(project_id, "completed")

            logger.info(f"Successfully generated project {project_id}")

        except Exception as e:
            logger.error(f"Error generating project {project_id}: {str(e)}")
            await self._update_project_status(project_id, "failed", str(e))
            raise

    def _extract_project_config(
        self,
        workflow_data: Dict[str, Any],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract project configuration from workflow data"""
        # Default configuration
        config = {
            "project_name": "fastapi_app",
            "description": "Generated FastAPI application",
            "version": "1.0.0",
            "author": "Dev-conditional",
            "database": "postgresql",
            "authentication": True,
            "cors": True,
            "docs": True,
            "endpoints": [],
            "models": [],
            "dependencies": [
                "fastapi>=0.104.1",
                "uvicorn[standard]>=0.24.0",
                "pydantic>=2.5.0",
                "sqlalchemy>=2.0.23",
                "psycopg2-binary>=2.9.9"
            ]
        }

        # Extract from workflow nodes
        if "nodes" in workflow_data:
            for node in workflow_data["nodes"]:
                if node.get("type") == "api_endpoint":
                    config["endpoints"].append(self._extract_endpoint_config(node))
                elif node.get("type") == "database_model":
                    config["models"].append(self._extract_model_config(node))
                elif node.get("type") == "auth_config":
                    config.update(self._extract_auth_config(node))

        # Merge custom configuration
        if custom_config:
            config.update(custom_config)

        return config

    def _extract_endpoint_config(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract API endpoint configuration from node"""
        return {
            "path": node.get("data", {}).get("path", "/"),
            "method": node.get("data", {}).get("method", "GET"),
            "function_name": node.get("data", {}).get("function_name", "get_items"),
            "description": node.get("data", {}).get("description", ""),
            "parameters": node.get("data", {}).get("parameters", []),
            "response_model": node.get("data", {}).get("response_model", None)
        }

    def _extract_model_config(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract database model configuration from node"""
        return {
            "name": node.get("data", {}).get("name", "Item"),
            "fields": node.get("data", {}).get("fields", []),
            "relationships": node.get("data", {}).get("relationships", [])
        }

    def _extract_auth_config(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract authentication configuration from node"""
        return {
            "authentication": node.get("data", {}).get("enabled", True),
            "auth_type": node.get("data", {}).get("type", "jwt"),
            "oauth_providers": node.get("data", {}).get("oauth_providers", [])
        }

    async def _generate_project_structure(
        self,
        config: Dict[str, Any],
        template_name: str
    ) -> Dict[str, Any]:
        """Generate project structure based on configuration"""
        structure = {
            "files": {},
            "directories": ["src", "tests", "docs"]
        }

        # Main application file
        structure["files"]["src/main.py"] = await self._render_template(
            f"{template_name}/main.py.jinja2", config
        )

        # Models
        if config["models"]:
            structure["files"]["src/models.py"] = await self._render_template(
                f"{template_name}/models.py.jinja2", config
            )

        # Schemas
        if config["models"]:
            structure["files"]["src/schemas.py"] = await self._render_template(
                f"{template_name}/schemas.py.jinja2", config
            )

        # Database
        structure["files"]["src/database.py"] = await self._render_template(
            f"{template_name}/database.py.jinja2", config
        )

        # Routes
        if config["endpoints"]:
            structure["files"]["src/routes.py"] = await self._render_template(
                f"{template_name}/routes.py.jinja2", config
            )

        # Configuration
        structure["files"]["src/config.py"] = await self._render_template(
            f"{template_name}/config.py.jinja2", config
        )

        # Requirements
        structure["files"]["requirements.txt"] = "\\n".join(config["dependencies"])

        # README
        structure["files"]["README.md"] = await self._render_template(
            f"{template_name}/README.md.jinja2", config
        )

        # Docker
        structure["files"]["Dockerfile"] = await self._render_template(
            f"{template_name}/Dockerfile.jinja2", config
        )

        return structure

    async def _render_template(self, template_path: str, config: Dict[str, Any]) -> str:
        """Render a Jinja2 template"""
        try:
            template = self.jinja_env.get_template(template_path)
            return template.render(**config)
        except Exception as e:
            logger.error(f"Error rendering template {template_path}: {str(e)}")
            # Fallback to basic template
            return f"# Generated from {template_path}\\n# Template rendering failed: {str(e)}"

    async def _generate_files(
        self,
        project_dir: str,
        structure: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actual files in project directory"""
        generated_files = {}

        for file_path, content in structure["files"].items():
            full_path = os.path.join(project_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, 'w') as f:
                f.write(content)

            generated_files[file_path] = {
                "path": full_path,
                "size": len(content),
                "type": "file"
            }

        return generated_files

    async def _create_project_zip(self, project_dir: str, project_id: int) -> str:
        """Create ZIP file of generated project"""
        zip_path = os.path.join(settings.GENERATED_PROJECTS_DIR, f"project_{project_id}.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    zipf.write(file_path, arcname)

        return zip_path

    async def _update_project_status(
        self,
        project_id: int,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update project status in database"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, update

            query = select(GeneratedProject).where(GeneratedProject.id == project_id)
            result = await session.execute(query)
            project = result.scalar_one_or_none()

            if project:
                update_data = {"generated_code": {"status": status}}
                if error_message:
                    update_data["generated_code"]["error"] = error_message

                stmt = (
                    update(GeneratedProject)
                    .where(GeneratedProject.id == project_id)
                    .values(**update_data)
                )
                await session.execute(stmt)
                await session.commit()

    async def _save_project_result(
        self,
        project_id: int,
        structure: Dict[str, Any],
        zip_path: str,
        template_name: str
    ):
        """Save project generation result to database"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, update

            query = select(GeneratedProject).where(GeneratedProject.id == project_id)
            result = await session.execute(query)
            project = result.scalar_one_or_none()

            if project:
                project.generated_code = structure
                project.file_path = zip_path
                project.template_used = template_name

                await session.commit()

    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        """List available code generation templates"""
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates")
        templates = []

        if os.path.exists(templates_dir):
            for template_name in os.listdir(templates_dir):
                template_path = os.path.join(templates_dir, template_name)
                if os.path.isdir(template_path):
                    templates.append({
                        "name": template_name,
                        "description": f"{template_name} template",
                        "path": template_path
                    })

        return templates

    @classmethod
    def get_template_info(cls, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific template"""
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates")
        template_path = os.path.join(templates_dir, template_name)

        if os.path.exists(template_path):
            return {
                "name": template_name,
                "path": template_path,
                "files": os.listdir(template_path) if os.path.isdir(template_path) else []
            }

        return None

    @classmethod
    def preview_generation(
        cls,
        workflow_data: Dict[str, Any],
        template_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Preview what would be generated"""
        generator = cls()
        config = generator._extract_project_config(workflow_data, custom_config)

        return {
            "config": config,
            "estimated_files": len(config["models"]) + len(config["endpoints"]) + 5,
            "template": template_name,
            "features": [
                "FastAPI application",
                "SQLAlchemy models" if config["models"] else None,
                "Pydantic schemas" if config["models"] else None,
                "API endpoints" if config["endpoints"] else None,
                "Database integration",
                "Docker support",
                "Documentation"
            ]
        }

    @classmethod
    def import_project(cls, zip_path: str) -> Dict[str, Any]:
        """Import and analyze existing project"""
        project_info = {
            "name": "imported_project",
            "type": "fastapi",
            "files": [],
            "dependencies": [],
            "structure": {}
        }

        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                project_info["files"] = file_list

                # Look for requirements.txt
                if "requirements.txt" in file_list:
                    with zipf.open("requirements.txt") as f:
                        content = f.read().decode('utf-8')
                        project_info["dependencies"] = [
                            line.strip() for line in content.split('\\n')
                            if line.strip() and not line.startswith('#')
                        ]

                # Analyze Python files
                python_files = [f for f in file_list if f.endswith('.py')]
                project_info["python_files"] = python_files
                project_info["structure"] = cls._analyze_project_structure(python_files, zipf)

        except Exception as e:
            logger.error(f"Error importing project: {str(e)}")
            project_info["error"] = str(e)

        return project_info

    @classmethod
    def _analyze_project_structure(cls, python_files: List[str], zipf) -> Dict[str, Any]:
        """Analyze project structure from Python files"""
        structure = {
            "has_fastapi": False,
            "has_sqlalchemy": False,
            "models": [],
            "routes": [],
            "main_file": None
        }

        for file_path in python_files:
            try:
                with zipf.open(file_path) as f:
                    content = f.read().decode('utf-8')

                # Simple AST analysis
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "fastapi":
                            structure["has_fastapi"] = True
                        elif node.module == "sqlalchemy":
                            structure["has_sqlalchemy"] = True

                    elif isinstance(node, ast.ClassDef):
                        if any("Model" in base.id for base in node.bases if isinstance(base, ast.Name)):
                            structure["models"].append(node.name)

                    elif isinstance(node, ast.FunctionDef) and "route" in file_path:
                        structure["routes"].append(node.name)

                if file_path.endswith("main.py"):
                    structure["main_file"] = file_path

            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {str(e)}")

        return structure