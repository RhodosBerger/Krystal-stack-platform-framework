"""
Django REST Framework Serializers
"""

from rest_framework import serializers
from .models import (
    Machine, Telemetry, Project, Tool, Job,
    InspectionReport, EconomicRecord, Alert,
    RiseUser, Organization
)

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = '__all__'

class RiseUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiseUser
        fields = ['id', 'username', 'email', 'role', 'organization', 'phone']
        extra_kwargs = {'password': {'write_only': True}}

class MachineSerializer(serializers.ModelSerializer):
    current_oee = serializers.SerializerMethodField()
    
    class Meta:
        model = Machine
        fields = '__all__'
    
    def get_current_oee(self, obj):
        try:
            return obj.get_current_oee()
        except:
            return None

class TelemetrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Telemetry
        fields = '__all__'

class ProjectSerializer(serializers.ModelSerializer):
    created_by_name = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = Project
        fields = '__all__'

class ToolSerializer(serializers.ModelSerializer):
    remaining_life = serializers.ReadOnlyField()
    life_percentage = serializers.ReadOnlyField()
    should_replace = serializers.ReadOnlyField()
    
    class Meta:
        model = Tool
        fields = '__all__'

class JobSerializer(serializers.ModelSerializer):
    project_name = serializers.CharField(source='project.name', read_only=True)
    machine_name = serializers.CharField(source='machine.name', read_only=True)
    progress_percentage = serializers.ReadOnlyField()
    
    class Meta:
        model = Job
        fields = '__all__'

class InspectionReportSerializer(serializers.ModelSerializer):
    inspector_name = serializers.CharField(source='inspector.username', read_only=True)
    
    class Meta:
        model = InspectionReport
        fields = '__all__'

class EconomicRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = EconomicRecord
        fields = '__all__'

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = '__all__'
