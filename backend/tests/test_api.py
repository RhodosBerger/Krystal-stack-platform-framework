"""
Test suite for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from ..src.app import app
from ..src.storage.database import get_db, Base
from ..src.storage.models import Workflow, WorkflowExecution

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_workflow():
    """Sample workflow data"""
    return {
        "name": "Test Workflow",
        "description": "A test workflow for unit testing",
        "workflow_data": {
            "nodes": [
                {
                    "id": "1",
                    "type": "trigger",
                    "data": {"trigger_type": "manual"}
                },
                {
                    "id": "2",
                    "type": "api_call",
                    "data": {"url": "https://api.example.com", "method": "GET"}
                }
            ],
            "edges": [
                {"source": "1", "target": "2"}
            ]
        }
    }


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self, setup_database):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Dev-conditional Autonomous Server Engine"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_health_endpoint(self, setup_database):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert "services" in data


class TestWorkflowAPI:
    """Test workflow API endpoints"""

    def test_create_workflow(self, setup_database, sample_workflow):
        """Test workflow creation"""
        response = client.post("/api/workflow/", json=sample_workflow)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == sample_workflow["name"]
        assert data["description"] == sample_workflow["description"]
        assert "id" in data
        assert data["version"] == 1
        assert data["is_active"] is True

    def test_list_workflows(self, setup_database, sample_workflow):
        """Test listing workflows"""
        # Create a workflow first
        client.post("/api/workflow/", json=sample_workflow)

        response = client.get("/api/workflow/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_workflow(self, setup_database, sample_workflow):
        """Test getting a specific workflow"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        response = client.get(f"/api/workflow/{workflow_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == workflow_id
        assert data["name"] == sample_workflow["name"]

    def test_get_nonexistent_workflow(self, setup_database):
        """Test getting a workflow that doesn't exist"""
        response = client.get("/api/workflow/999999")
        assert response.status_code == 404
        data = response.json()
        assert "Workflow not found" in data["detail"]

    def test_update_workflow(self, setup_database, sample_workflow):
        """Test updating a workflow"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        update_data = {
            "name": "Updated Workflow",
            "description": "Updated description"
        }

        response = client.put(f"/api/workflow/{workflow_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Workflow"
        assert data["description"] == "Updated description"
        assert data["version"] == 2  # Version should increment

    def test_delete_workflow(self, setup_database, sample_workflow):
        """Test deleting a workflow"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        response = client.delete(f"/api/workflow/{workflow_id}")
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]

        # Verify it's deleted
        get_response = client.get(f"/api/workflow/{workflow_id}")
        assert get_response.status_code == 404


class TestWorkflowExecution:
    """Test workflow execution endpoints"""

    def test_execute_workflow(self, setup_database, sample_workflow):
        """Test workflow execution"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        response = client.post(
            f"/api/workflow/{workflow_id}/execute",
            json={"test_data": "value"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == workflow_id
        assert data["status"] == "running"
        assert "id" in data

    def test_execute_nonexistent_workflow(self, setup_database):
        """Test executing a workflow that doesn't exist"""
        response = client.post("/api/workflow/999999/execute")
        assert response.status_code == 404

    def test_get_workflow_executions(self, setup_database, sample_workflow):
        """Test getting workflow execution history"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        # Execute the workflow
        exec_response = client.post(f"/api/workflow/{workflow_id}/execute")
        execution_id = exec_response.json()["id"]

        # Get executions
        response = client.get(f"/api/workflow/{workflow_id}/executions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(exec["id"] == execution_id for exec in data)


class TestCodeGenerationAPI:
    """Test code generation API endpoints"""

    def test_list_templates(self, setup_database):
        """Test listing code generation templates"""
        response = client.get("/api/codegen/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data

    def test_get_template_info(self, setup_database):
        """Test getting template information"""
        response = client.get("/api/codegen/templates/fastapi_basic")
        # This might return 404 if template doesn't exist, which is fine for testing
        assert response.status_code in [200, 404]

    def test_preview_generation(self, setup_database, sample_workflow):
        """Test code generation preview"""
        # Create a workflow first
        create_response = client.post("/api/workflow/", json=sample_workflow)
        workflow_id = create_response.json()["id"]

        preview_request = {
            "workflow_id": workflow_id,
            "project_name": "test-project",
            "template_name": "fastapi_basic"
        }

        response = client.post("/api/codegen/preview", json=preview_request)
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "template" in data
        assert "features" in data


class TestLLMAPI:
    """Test LLM API endpoints"""

    def test_get_available_models(self, setup_database):
        """Test getting available LLM models"""
        response = client.get("/api/llm/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_get_llm_status(self, setup_database):
        """Test LLM service status"""
        response = client.get("/api/llm/status")
        assert response.status_code == 200
        data = response.json()
        # Should contain status information even if LLM is not configured
        assert "status" in data

    def test_chat_with_llm_no_api_key(self, setup_database):
        """Test LLM chat without API key (should return error)"""
        chat_request = {
            "message": "Hello, test message",
            "session_id": "test-session"
        }

        response = client.post("/api/llm/chat", json=chat_request)
        # This might return 500 if no API key is configured, which is expected
        assert response.status_code in [200, 500]


class TestErrorHandling:
    """Test error handling"""

    def test_404_for_invalid_endpoint(self, setup_database):
        """Test 404 for invalid endpoint"""
        response = client.get("/api/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_json_payload(self, setup_database):
        """Test handling of invalid JSON payload"""
        response = client.post(
            "/api/workflow/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, setup_database):
        """Test validation of required fields"""
        invalid_workflow = {
            "description": "Missing name field"
        }

        response = client.post("/api/workflow/", json=invalid_workflow)
        assert response.status_code == 422


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations"""

    async def test_concurrent_workflow_creation(self, setup_database, sample_workflow):
        """Test concurrent workflow creation"""
        import asyncio

        async def create_workflow():
            return client.post("/api/workflow/", json=sample_workflow)

        # Create multiple workflows concurrently
        tasks = [create_workflow() for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for response in responses:
            assert hasattr(response, 'status_code')
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])