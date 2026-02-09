from fastapi import APIRouter
from . import telemetry_routes, machine_routes

router = APIRouter()

# Include all route modules
router.include_router(telemetry_routes.router, prefix="/telemetry", tags=["telemetry"])
router.include_router(machine_routes.router, prefix="/machines", tags=["machines"])