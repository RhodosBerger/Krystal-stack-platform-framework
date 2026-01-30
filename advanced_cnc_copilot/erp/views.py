"""
Django REST Framework Views
API endpoints combining Flask + Django
"""

from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.utils import timezone
from django.db import models
from datetime import timedelta
import requests

from .models import (
    Machine, Telemetry, Project, Tool, Job, 
    InspectionReport, EconomicRecord, Alert, ConfigurationProfile
)
from .serializers import (
    MachineSerializer, TelemetrySerializer, ProjectSerializer,
    ToolSerializer, JobSerializer, AlertSerializer
)

# Flask microservice URL
FLASK_SERVICE_URL = 'http://localhost:5000'

# ====================
# MACHINE MANAGEMENT
# ====================

class MachineViewSet(viewsets.ModelViewSet):
    """Machine CRUD + telemetry integration"""
    queryset = Machine.objects.all()
    serializer_class = MachineSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Filter by user's organization
        return Machine.objects.filter(
            organization=self.request.user.organization
        )
    
    @action(detail=True, methods=['get'])
    def telemetry(self, request, pk=None):
        """Get real-time telemetry from Flask service"""
        try:
            response = requests.get(f'{FLASK_SERVICE_URL}/api/telemetry/current')
            data = response.json()
            
            # Save to database
            Telemetry.objects.create(
                machine_id=pk,
                timestamp=timezone.now(),
                rpm=data.get('rpm', 0),
                load=data.get('load', 0),
                vibration_x=data.get('vibration', {}).get('x', 0),
                vibration_y=data.get('vibration', {}).get('y', 0),
                vibration_z=data.get('vibration', {}).get('z', 0),
                spindle_temp=data.get('spindle_temp', 0),
                position_x=data.get('position', {}).get('x', 0),
                position_y=data.get('position', {}).get('y', 0),
                position_z=data.get('position', {}).get('z', 0),
                dopamine=data.get('dopamine', 50),
                cortisol=data.get('cortisol', 50),
                signal=data.get('signal', 'GREEN'),
                active_tool=data.get('tool_id', 'T01')
            )
            
            return Response(data)
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    
    @action(detail=True, methods=['get'])
    def oee(self, request, pk=None):
        """Calculate OEE for machine"""
        machine = self.get_object()
        oee_data = machine.get_current_oee()
        return Response(oee_data)
    
    @action(detail=True, methods=['post'])
    def dopamine_check(self, request, pk=None):
        """Check dopamine score via Flask"""
        try:
            response = requests.post(
                f'{FLASK_SERVICE_URL}/api/dopamine/evaluate',
                json=request.data
            )
            return Response(response.json())
        except Exception as e:
            return Response({'error': str(e)}, status=500)

# ====================
# PROJECT MANAGEMENT
# ====================

class ProjectViewSet(viewsets.ModelViewSet):
    """Project CRUD + LLM suggestions"""
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    
    @action(detail=True, methods=['get'])
    def similar(self, request, pk=None):
        """Find similar projects"""
        project = self.get_object()
        similar = project.get_similar_projects(limit=10)
        serializer = self.get_serializer(similar, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def suggest_params(self, request, pk=None):
        """Get LLM parameter suggestions"""
        project = self.get_object()
        
        # Call Protocol Conductor
        from cms.protocol_conductor import ProtocolConductor
        conductor = ProtocolConductor()
        
        context = {
            'material': project.material,
            'complexity': project.complexity_score,
            'dimensions': project.dimensions,
            'similar_projects': [p.llm_suggestions for p in project.get_similar_projects(3)]
        }
        
        suggestion = conductor.suggest_parameters(context)
        
        # Save suggestion
        project.llm_suggestions = suggestion
        project.save()
        
        return Response(suggestion)

# ====================
# TOOL MANAGEMENT
# ====================

class ToolViewSet(viewsets.ModelViewSet):
    """Tool inventory management"""
    queryset = Tool.objects.all()
    serializer_class = ToolSerializer
    
    @action(detail=False, methods=['get'])
    def needs_replacement(self, request):
        """List tools needing replacement"""
        tools = Tool.objects.filter(
            current_usage__gte=0.9 * models.F('expected_life')
        )
        serializer = self.get_serializer(tools, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def log_usage(self, request, pk=None):
        """Update tool usage time"""
        tool = self.get_object()
        minutes = request.data.get('minutes', 0)
        
        tool.current_usage += minutes
        if tool.current_usage >= tool.expected_life:
            tool.status = 'WORN'
            
            # Create alert
            Alert.objects.create(
                alert_type='TOOL_REPLACEMENT',
                priority='HIGH',
                message=f'Tool {tool.tool_id} ({tool.type} Ø{tool.diameter}mm) needs replacement!',
            )
        
        tool.save()
        return Response(self.get_serializer(tool).data)

# ====================
# JOB SCHEDULING
# ====================

class JobViewSet(viewsets.ModelViewSet):
    """Production job management"""
    queryset = Job.objects.all()
    serializer_class = JobSerializer
    
    @action(detail=False, methods=['get'])
    def schedule(self, request):
        """Get optimized production schedule"""
        from cms.production_scheduler import ProductionScheduler
        
        scheduler = ProductionScheduler()
        queued_jobs = Job.objects.filter(status='QUEUED')
        machines = Machine.objects.filter(is_active=True)
        
        schedule = scheduler.optimize_schedule(
            list(queued_jobs), 
            list(machines)
        )
        
        return Response(schedule)
    
    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start a job"""
        job = self.get_object()
        job.status = 'IN_PROGRESS'
        job.started_at = timezone.now()
        job.save()
        
        return Response({'message': 'Job started', 'job_id': job.job_id})
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Complete a job"""
        job = self.get_object()
        job.status = 'COMPLETED'
        job.completed_at = timezone.now()
        job.completed_quantity = job.quantity
        job.save()
        
        # Calculate economics
        from .economics import EconomicsCalculator
        calc = EconomicsCalculator(job)
        economic_record = calc.create_record()
        
        return Response({
            'message': 'Job completed',
            'economics': {
                'total_cost': float(economic_record.total_cost),
                'cost_per_part': float(economic_record.cost_per_part),
                'savings': float(economic_record.savings) if economic_record.savings else 0
            }
        })

# ====================
# CONFIGURATION (Dynamic Forms)
# ====================

@api_view(['GET'])
def get_form_config(request, config_type):
    """Get dynamic form configuration"""
    from cms.dynamic_form_builder import DynamicFormBuilder, create_safety_config
    
    # Example: safety config
    if config_type == 'safety':
        builder = create_safety_config()
        config = builder.build_config()
        return Response(config)
    
    return Response({'error': 'Config type not found'}, status=404)

@api_view(['POST'])
def save_form_config(request, config_type):
    """Save dynamic form values"""
    values = request.data
    
    # Validate
    from cms.dynamic_form_builder import create_safety_config
    builder = create_safety_config()
    errors = builder.validate(values)
    
    if errors:
        return Response({'errors': errors}, status=400)
    
    # Save to database
    profile, created = ConfigurationProfile.objects.update_or_create(
        profile_type=config_type,
        organization=request.user.organization,
        defaults={'values': values}
    )
    
    return Response({
        'message': 'Configuration saved',
        'profile_id': profile.id
    })

# ====================
# ANALYTICS
# ====================

@api_view(['GET'])
def get_dashboard_analytics(request):
    """Get comprehensive dashboard data"""
    
    # Machines
    machines = Machine.objects.filter(organization=request.user.organization, is_active=True)
    machine_count = machines.count()
    
    # Jobs
    jobs_in_progress = Job.objects.filter(status='IN_PROGRESS').count()
    jobs_queued = Job.objects.filter(status='QUEUED').count()
    
    # OEE average
    oees = [m.get_current_oee()['oee'] for m in machines]
    avg_oee = sum(oees) / len(oees) if oees else 0
    
    # Alerts
    unacknowledged_alerts = Alert.objects.filter(acknowledged=False).count()
    
    # Tools
    tools_critical = Tool.objects.filter(
        current_usage__gte=0.9 * models.F('expected_life')
    ).count()
    
    # Economics
    today_jobs = Job.objects.filter(
        completed_at__date=timezone.now().date()
    )
    today_revenue = sum([
        float(EconomicRecord.objects.get(job=job).total_cost) 
        for job in today_jobs if hasattr(job, 'economicrecord')
    ])
    
    return Response({
        'machines': {
            'total': machine_count,
            'avg_oee': round(avg_oee, 2)
        },
        'jobs': {
            'in_progress': jobs_in_progress,
            'queued': jobs_queued
        },
        'alerts': {
            'unacknowledged': unacknowledged_alerts
        },
        'tools': {
            'critical': tools_critical
        },
        'economics': {
            'today_revenue': today_revenue
        }
    })
