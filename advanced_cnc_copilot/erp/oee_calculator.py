"""
OEE Calculator Implementation
"""

from datetime import datetime, timedelta
from django.utils import timezone
from django.db.models import Sum, Count, Q

class OEECalculator:
    """Calculate Overall Equipment Effectiveness"""
    
    def __init__(self, machine):
        self.machine = machine
        self.shift_start = self.get_shift_start()
        self.shift_end = timezone.now()
        self.planned_time = 8 * 60  # 8 hour shift in minutes
    
    def get_shift_start(self):
        """Get start of current shift"""
        now = timezone.now()
        # Day shift: 08:00-16:00
        shift_start = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now.hour < 8:
            shift_start -= timedelta(days=1)
        return shift_start
    
    def calculate_availability(self):
        """Availability = Operating Time / Planned Time"""
        from .models import Job
        
        # Get jobs completed during shift
        jobs = Job.objects.filter(
            machine=self.machine,
            started_at__gte=self.shift_start,
            status__in=['IN_PROGRESS', 'COMPLETED']
        )
        
        # Calculate operating time
        operating_time = 0
        for job in jobs:
            if job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds() / 60
            else:
                duration = (self.shift_end - job.started_at).total_seconds() / 60
            operating_time += duration
        
        availability = (operating_time / self.planned_time) * 100 if self.planned_time > 0 else 0
        return min(100, availability)
    
    def calculate_performance(self):
        """Performance = Actual Output / Target Output"""
        from .models import Job
        
        # Get completed parts
        jobs = Job.objects.filter(
            machine=self.machine,
            started_at__gte=self.shift_start
        )
        
        actual_output = sum([job.completed_quantity for job in jobs])
        
        # Target based on estimated cycle times
        target_output = 0
        for job in jobs:
            if job.project.estimated_cycle_time > 0:
                target_for_job = (self.planned_time / job.project.estimated_cycle_time)
                target_output += target_for_job
        
        performance = (actual_output / target_output) * 100 if target_output > 0 else 0
        return min(100, performance)
    
    def calculate_quality(self):
        """Quality = Good Units / Total Units"""
        from .models import Job, InspectionReport
        
        jobs = Job.objects.filter(
            machine=self.machine,
            started_at__gte=self.shift_start,
            status='COMPLETED'
        )
        
        total_units = sum([job.completed_quantity for job in jobs])
        
        # Count rejected units from inspection reports
        rejected = 0
        for job in jobs:
            failed_inspections = InspectionReport.objects.filter(
                job=job,
                passed=False
            ).count()
            rejected += failed_inspections
        
        good_units = total_units - rejected
        quality = (good_units / total_units) * 100 if total_units > 0 else 100
        return quality
    
    def get_oee(self):
        """Calculate overall OEE"""
        availability = self.calculate_availability()
        performance = self.calculate_performance()
        quality = self.calculate_quality()
        
        oee = (availability / 100) * (performance / 100) * (quality / 100) * 100
        
        # Classification
        if oee >= 85:
            classification = "World Class"
        elif oee >= 60:
            classification = "Good"
        else:
            classification = "Needs Improvement"
        
        return {
            "oee": round(oee, 2),
            "availability": round(availability, 2),
            "performance": round(performance, 2),
            "quality": round(quality, 2),
            "classification": classification,
            "shift_start": self.shift_start.isoformat(),
            "shift_end": self.shift_end.isoformat()
        }
