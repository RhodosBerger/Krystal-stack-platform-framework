"""
LLM Integration Service for Dev-conditional Server Engine
Provides AI assistance for workflow design, code generation, and validation
"""

import httpx
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json

from ..config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Main LLM integration service"""

    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=settings.OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        self.model = settings.OPENAI_MODEL

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send chat message to LLM and get response"""
        try:
            # Construct system prompt based on context
            system_prompt = self._get_system_prompt(context)

            # Create conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            # Add context as additional messages if provided
            if context and "conversation_history" in context:
                messages[1:1] = context["conversation_history"][-5:]  # Last 5 messages

            # Make API call
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")

            data = response.json()
            llm_response = data["choices"][0]["message"]["content"]

            return {
                "response": llm_response,
                "session_id": session_id,
                "context": context,
                "usage": data.get("usage", {})
            }

        except Exception as e:
            logger.error(f"LLM chat error: {str(e)}")
            raise

    async def suggest_workflow(
        self,
        user_goal: str,
        current_workflow: Dict[str, Any],
        current_step: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest workflow improvements and next steps"""
        try:
            prompt = self._build_workflow_suggestion_prompt(user_goal, current_workflow, current_step)

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert workflow designer for automation systems. Provide specific, actionable suggestions for improving workflows."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code}")

            data = response.json()
            suggestions = data["choices"][0]["message"]["content"]

            # Parse suggestions into structured format
            parsed_suggestions = self._parse_workflow_suggestions(suggestions)

            return {
                "suggestions": parsed_suggestions,
                "raw_response": suggestions,
                "workflow_goal": user_goal
            }

        except Exception as e:
            logger.error(f"Workflow suggestion error: {str(e)}")
            raise

    async def validate_code(
        self,
        code: str,
        language: str,
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate code using LLM analysis"""
        try:
            prompt = self._build_code_validation_prompt(code, language, requirements)

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert code reviewer. Analyze code for correctness, security, and best practices."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code}")

            data = response.json()
            validation = data["choices"][0]["message"]["content"]

            # Parse validation result
            parsed_validation = self._parse_code_validation(validation)

            return {
                "validation": parsed_validation,
                "raw_response": validation,
                "language": language,
                "is_valid": parsed_validation.get("overall_score", 0) > 0.7
            }

        except Exception as e:
            logger.error(f"Code validation error: {str(e)}")
            raise

    async def optimize_code(
        self,
        code: str,
        language: str,
        optimization_type: str = "performance"
    ) -> Dict[str, Any]:
        """Optimize code using LLM suggestions"""
        try:
            prompt = self._build_code_optimization_prompt(code, language, optimization_type)

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert code optimizer. Focus on {optimization_type} improvements while maintaining functionality."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code}")

            data = response.json()
            optimization = data["choices"][0]["message"]["content"]

            # Parse optimization result
            parsed_optimization = self._parse_code_optimization(optimization)

            return {
                "optimization": parsed_optimization,
                "raw_response": optimization,
                "language": language,
                "optimization_type": optimization_type
            }

        except Exception as e:
            logger.error(f"Code optimization error: {str(e)}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """Get LLM service status"""
        try:
            # Simple health check
            response = await self.client.get("/models", timeout=5.0)

            if response.status_code == 200:
                return {
                    "status": "connected",
                    "model": self.model,
                    "base_url": settings.OPENAI_BASE_URL,
                    "has_api_key": bool(settings.OPENAI_API_KEY)
                }
            else:
                return {
                    "status": "error",
                    "error": f"API returned status {response.status_code}"
                }

        except Exception as e:
            return {
                "status": "disconnected",
                "error": str(e)
            }

    def _get_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Get system prompt based on context"""
        base_prompt = """You are Dev-conditional AI Assistant, an expert in building autonomous server applications,
workflow automation, and code generation. You help users create powerful server applications with
conditional logic, LLM integration, and autonomous execution capabilities.

Your expertise includes:
- FastAPI web application development
- Database design with SQLAlchemy
- Workflow automation and node-based systems
- Code generation and template systems
- Docker containerization
- API design and integration
- Security best practices

Be helpful, specific, and provide actionable advice. When suggesting code, ensure it follows best practices."""

        if context:
            if "current_task" in context:
                base_prompt += f"\n\nCurrent task: {context['current_task']}"
            if "project_type" in context:
                base_prompt += f"\n\nProject type: {context['project_type']}"

        return base_prompt

    def _build_workflow_suggestion_prompt(
        self,
        user_goal: str,
        current_workflow: Dict[str, Any],
        current_step: Optional[str]
    ) -> str:
        """Build prompt for workflow suggestions"""
        prompt = f"User Goal: {user_goal}\n\n"

        if current_workflow:
            prompt += f"Current Workflow: {json.dumps(current_workflow, indent=2)}\n\n"

        if current_step:
            prompt += f"Current Step: {current_step}\n\n"

        prompt += """Please provide specific suggestions to:
1. Improve the workflow structure
2. Add missing functionality
3. Optimize performance
4. Ensure error handling
5. Suggest best practices

Format your response as JSON with these keys:
- "improvements": list of specific improvements
- "missing_nodes": list of recommended node types to add
- "best_practices": list of best practice recommendations
- "next_steps": list of actionable next steps"""

        return prompt

    def _build_code_validation_prompt(
        self,
        code: str,
        language: str,
        requirements: Optional[List[str]]
    ) -> str:
        """Build prompt for code validation"""
        prompt = f"Language: {language}\n\n"
        prompt += f"Code to validate:\n```{language}\n{code}\n```\n\n"

        if requirements:
            prompt += f"Requirements:\n" + "\n".join(f"- {req}" for req in requirements) + "\n\n"

        prompt += """Please analyze this code for:
1. Correctness and functionality
2. Security vulnerabilities
3. Performance issues
4. Code style and readability
5. Best practices adherence

Format your response as JSON with:
- "overall_score": float between 0 and 1
- "issues": list of identified issues with severity levels
- "suggestions": list of improvement suggestions
- "security_concerns": list of security issues if any"""

        return prompt

    def _build_code_optimization_prompt(
        self,
        code: str,
        language: str,
        optimization_type: str
    ) -> str:
        """Build prompt for code optimization"""
        prompt = f"Language: {language}\n"
        prompt += f"Optimization focus: {optimization_type}\n\n"
        prompt += f"Code to optimize:\n```{language}\n{code}\n```\n\n"

        prompt += f"""Please optimize this code for {optimization_type}. Provide:
1. The optimized version of the code
2. Explanation of changes made
3. Performance improvements achieved (if applicable)
4. Any trade-offs or considerations

Format your response as JSON with:
- "optimized_code": the improved code
- "changes": list of changes made
- "improvements": description of improvements
- "trade_offs": any trade-offs to consider"""

        return prompt

    def _parse_workflow_suggestions(self, suggestions: str) -> Dict[str, Any]:
        """Parse workflow suggestions from LLM response"""
        try:
            # Try to parse as JSON first
            if suggestions.strip().startswith('{'):
                return json.loads(suggestions)
            else:
                # Extract structured info from text response
                return {
                    "raw_suggestions": suggestions,
                    "improvements": ["Review workflow structure"],
                    "missing_nodes": ["Add error handling"],
                    "best_practices": ["Implement logging"],
                    "next_steps": ["Test current workflow"]
                }
        except Exception:
            return {
                "raw_suggestions": suggestions,
                "improvements": [],
                "missing_nodes": [],
                "best_practices": [],
                "next_steps": []
            }

    def _parse_code_validation(self, validation: str) -> Dict[str, Any]:
        """Parse code validation from LLM response"""
        try:
            if validation.strip().startswith('{'):
                return json.loads(validation)
            else:
                return {
                    "raw_validation": validation,
                    "overall_score": 0.8,
                    "issues": [],
                    "suggestions": [validation],
                    "security_concerns": []
                }
        except Exception:
            return {
                "raw_validation": validation,
                "overall_score": 0.5,
                "issues": ["Unable to parse validation"],
                "suggestions": [],
                "security_concerns": []
            }

    def _parse_code_optimization(self, optimization: str) -> Dict[str, Any]:
        """Parse code optimization from LLM response"""
        try:
            if optimization.strip().startswith('{'):
                return json.loads(optimization)
            else:
                return {
                    "raw_optimization": optimization,
                    "optimized_code": optimization,
                    "changes": ["Applied optimizations"],
                    "improvements": "Code optimized for performance",
                    "trade_offs": "None"
                }
        except Exception:
            return {
                "raw_optimization": optimization,
                "optimized_code": code,
                "changes": [],
                "improvements": "",
                "trade_offs": ""
            }

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()