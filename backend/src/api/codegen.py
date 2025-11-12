"""
Code Generation API endpoints for Dev-conditional Server Engine
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import shutil
import os
import json
import zipfile
from datetime import datetime

from ..storage.database import get_db
from ..storage.models import GeneratedProject, Workflow
from ..codegen.generator import CodeGenerator
from ..config import settings

router = APIRouter()


# Pydantic models
class CodeGenerationRequest(BaseModel):
    workflow_id: int
    project_name: str
    description: Optional[str] = None
    template_name: str = "fastapi_basic"
    custom_config: Optional[Dict[str, Any]] = {}


class ProjectResponse(BaseModel):
    id: int
    workflow_id: int
    name: str
    description: Optional[str]
    project_type: str
    template_used: Optional[str]
    created_at: str
    download_count: int

    class Config:
        from_attributes = True


class GenerationStatusResponse(BaseModel):
    status: str  # generating, completed, failed
    message: str
    project_id: Optional[int] = None


@router.post("/generate", response_model=GenerationStatusResponse)
async def generate_project(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Generate code from workflow"""
    # Verify workflow exists
    query = select(Workflow).where(Workflow.id == request.workflow_id)
    result = await db.execute(query)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Create project record
    project = GeneratedProject(
        workflow_id=request.workflow_id,
        name=request.project_name,
        description=request.description,
        project_type="fastapi",  # Default for now
        template_used=request.template_name
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    # Generate code in background
    background_tasks.add_task(
        CodeGenerator.generate_project_async,
        project.id,
        workflow.workflow_data,
        request.template_name,
        request.custom_config
    )

    return GenerationStatusResponse(
        status="generating",
        message="Code generation started",
        project_id=project.id
    )


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all generated projects"""
    query = (
        select(GeneratedProject)
        .offset(skip)
        .limit(limit)
        .order_by(GeneratedProject.created_at.desc())
    )
    result = await db.execute(query)
    projects = result.scalars().all()
    return projects


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific generated project"""
    query = select(GeneratedProject).where(GeneratedProject.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Increment download count if this is a download request
    # (This should be done in the download endpoint)

    return project


@router.get("/projects/{project_id}/files")
async def get_project_files(
    project_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get file structure of a generated project"""
    query = select(GeneratedProject).where(GeneratedProject.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.generated_code:
        raise HTTPException(status_code=400, detail="Project not generated yet")

    return {
        "files": project.generated_code.get("files", {}),
        "structure": project.generated_code.get("structure", {})
    }


@router.get("/projects/{project_id}/download")
async def download_project(
    project_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Download a generated project as ZIP file"""
    query = select(GeneratedProject).where(GeneratedProject.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.file_path:
        raise HTTPException(status_code=400, detail="Project files not available")

    # Increment download count
    project.download_count += 1
    await db.commit()

    # Create zip file if it doesn't exist
    zip_path = project.file_path.replace(".zip", "_temp.zip")
    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            project_dir = project.file_path.replace(".zip", "")
            if os.path.exists(project_dir):
                for root, dirs, files in os.walk(project_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, project_dir)
                        zipf.write(file_path, arcname)

    from fastapi.responses import FileResponse
    return FileResponse(
        path=zip_path,
        filename=f"{project.name}.zip",
        media_type="application/zip"
    )


@router.get("/templates")
async def list_templates():
    """List available code generation templates"""
    templates = CodeGenerator.list_templates()
    return {"templates": templates}


@router.get("/templates/{template_name}")
async def get_template_info(template_name: str):
    """Get information about a specific template"""
    template_info = CodeGenerator.get_template_info(template_name)
    if not template_info:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_info


@router.post("/preview")
async def preview_code_generation(
    request: CodeGenerationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Preview what code would be generated without actually generating it"""
    # Verify workflow exists
    query = select(Workflow).where(Workflow.id == request.workflow_id)
    result = await db.execute(query)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Generate preview
    preview = CodeGenerator.preview_generation(
        workflow.workflow_data,
        request.template_name,
        request.custom_config
    )

    return preview


@router.post("/import")
async def import_project(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Import an existing project"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    # Save uploaded file
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract and analyze project
        project_info = CodeGenerator.import_project(temp_path)
        return project_info
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)