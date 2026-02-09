"""
Django Integration for LLM Intelligence
Connects LLM system to Django CMS
"""

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.core.cache import cache

from backend.cms.llm_connector import LLMConnector, LLMConfig
from backend.cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint


class LLMQueryLog(models.Model):
    """
    Log all LLM queries for cost tracking and analysis
    """
    query = models.TextField()
    response = models.TextField()
    provider = models.CharField(max_length=50)
    model = models.CharField(max_length=100)
    tokens_used = models.IntegerField(null=True, blank=True)
    cost_estimate = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    response_time_seconds = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['provider', 'model']),
        ]
    
    def __str__(self):
        return f"{self.query[:50]}... ({self.created_at})"


class ManufacturingInsight(models.Model):
    """
    Store LLM-generated insights
    """
    PRIORITY_CHOICES = [
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    confidence_score = models.FloatField()
    data_points_analyzed = models.IntegerField()
    time_period_days = models.IntegerField()
    
    # Recommendations
    recommended_action = models.TextField()
    expected_benefit = models.TextField(blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
                                       null=True, blank=True, related_name='acknowledged_insights')
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at', 'priority']),
            models.Index(fields=['acknowledged']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.priority})"


class LLMIntelligenceService:
    """
    Service layer for LLM intelligence in Django
    Singleton pattern for efficiency
    """
    
    _instance = None
    _intelligence = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._intelligence = CrossSessionIntelligence()
        return cls._instance
    
    @classmethod
    def get_intelligence(cls) -> CrossSessionIntelligence:
        """Get cross-session intelligence instance"""
        if cls._intelligence is None:
            cls._intelligence = CrossSessionIntelligence()
        return cls._intelligence
    
    @classmethod
    def ask_question(cls, question: str, user=None) -> dict:
        """
        Ask LLM question and log it
        
        Args:
            question: Question to ask
            user: Django user making the request
        
        Returns:
            {
                'answer': str,
                'query_id': int,
                'response_time': float
            }
        """
        import time
        
        intelligence = cls.get_intelligence()
        
        start_time = time.time()
        answer = intelligence.ask_question(question)
        response_time = time.time() - start_time
        
        # Log query
        log = LLMQueryLog.objects.create(
            query=question,
            response=answer,
            provider=intelligence.llm.config.provider,
            model=intelligence.llm.config.model,
            response_time_seconds=response_time,
            user=user
        )
        
        return {
            'answer': answer,
            'query_id': log.id,
            'response_time': response_time
        }
    
    @classmethod
    def generate_insights(cls, days: int = 7) -> list:
        """
        Generate insights and save to database
        
        Args:
            days: Number of days to analyze
        
        Returns:
            List of created ManufacturingInsight objects
        """
        intelligence = cls.get_intelligence()
        
        # Generate report
        report = intelligence.generate_insights_report(days=days)
        
        created_insights = []
        
        # Save insights to database
        for insight_text in report.get('key_insights', []):
            insight = ManufacturingInsight.objects.create(
                title=insight_text[:200],  # First 200 chars as title
                description=insight_text,
                priority='medium',  # Would use LLM to determine
                confidence_score=0.8,  # Would get from LLM
                data_points_analyzed=report.get('data_points_analyzed', 0),
                time_period_days=days,
                recommended_action='Review and act on insight'
            )
            created_insights.append(insight)
        
        return created_insights
    
    @classmethod
    def predict_event(cls, event_type: str, current_indicators: dict) -> dict:
        """
        Predict future event
        
        Args:
            event_type: Type of event to predict
            current_indicators: Current sensor/process data
        
        Returns:
            Prediction dictionary
        """
        intelligence = cls.get_intelligence()
        return intelligence.predict_future_event(event_type, current_indicators)
    
    @classmethod
    def add_session_data(cls, session_id: str, data_type: str, 
                        data: dict, machine_id: str = None):
        """
        Add data point to cross-session intelligence
        
        Args:
            session_id: Session identifier
            data_type: Type of data
            data: Data dictionary
            machine_id: Machine identifier
        """
        intelligence = cls.get_intelligence()
        
        data_point = DataPoint(
            session_id=session_id,
            timestamp=timezone.now(),
            data_type=data_type,
            data=data,
            machine_id=machine_id
        )
        
        intelligence.add_data_point(data_point)
    
    @classmethod
    def get_cached_insights(cls, days: int = 7, cache_minutes: int = 30) -> list:
        """
        Get insights with caching
        
        Args:
            days: Days to analyze
            cache_minutes: Cache duration
        
        Returns:
            List of insights
        """
        cache_key = f'llm_insights_{days}d'
        
        insights = cache.get(cache_key)
        
        if insights is None:
            insights = cls.generate_insights(days=days)
            cache.set(cache_key, insights, timeout=cache_minutes * 60)
        
        return insights


# Django management command
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    """
    Management command: python manage.py generate_llm_insights
    """
    help = 'Generate LLM insights from manufacturing data'
    
    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=7, help='Days to analyze')
    
    def handle(self, *args, **options):
        days = options['days']
        
        self.stdout.write(f"Generating insights for last {days} days...")
        
        service = LLMIntelligenceService()
        insights = service.generate_insights(days=days)
        
        self.stdout.write(self.style.SUCCESS(
            f"Generated {len(insights)} insights"
        ))
        
        for insight in insights:
            self.stdout.write(f"  • {insight.title}")


# Django views integration
from django.http import JsonResponse
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin

class LLMAskView(LoginRequiredMixin, View):
    """
    View for asking LLM questions
    
    POST /api/llm/ask
    {
        "question": "Why did quality drop?"
    }
    """
    
    def post(self, request):
        import json
        
        try:
            data = json.loads(request.body)
            question = data.get('question')
            
            if not question:
                return JsonResponse({'error': 'Question required'}, status=400)
            
            result = LLMIntelligenceService.ask_question(question, user=request.user)
            
            return JsonResponse(result)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class LLMInsightsView(LoginRequiredMixin, View):
    """
    View for getting LLM insights
    
    GET /api/llm/insights?days=7
    """
    
    def get(self, request):
        days = int(request.GET.get('days', 7))
        
        insights = LLMIntelligenceService.get_cached_insights(days=days)
        
        return JsonResponse({
            'insights': [
                {
                    'id': i.id,
                    'title': i.title,
                    'description': i.description,
                    'priority': i.priority,
                    'confidence': i.confidence_score,
                    'recommended_action': i.recommended_action
                }
                for i in insights
            ],
            'count': len(insights)
        })


class LLMPredictView(LoginRequiredMixin, View):
    """
    View for event prediction
    
    POST /api/llm/predict
    {
        "event_type": "tool_failure",
        "current_indicators": {"vibration": 2.5, "load": 85}
    }
    """
    
    def post(self, request):
        import json
        
        try:
            data = json.loads(request.body)
            event_type = data.get('event_type')
            indicators = data.get('current_indicators', {})
            
            if not event_type:
                return JsonResponse({'error': 'event_type required'}, status=400)
            
            prediction = LLMIntelligenceService.predict_event(event_type, indicators)
            
            return JsonResponse(prediction)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


# URLs configuration
from django.urls import path

urlpatterns = [
    path('api/llm/ask', LLMAskView.as_view(), name='llm_ask'),
    path('api/llm/insights', LLMInsightsView.as_view(), name='llm_insights'),
    path('api/llm/predict', LLMPredictView.as_view(), name='llm_predict'),
]
