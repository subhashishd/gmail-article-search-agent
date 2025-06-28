"""
Workflow Orchestration Agent

This agent orchestrates complex workflows across multiple agents:
- Manages task delegation and coordination
- Handles inter-agent communication
- Monitors workflow progress and error handling
- Provides workflow analytics and optimization

Key Workflows:
- Article Processing Workflow (Gmail -> Content Analysis -> Storage)
- Search Workflow (Query -> Enhancement -> RAG -> Results)
- Content Enhancement Workflow (Analysis -> Classification -> Summarization)
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentMessage, AgentResponse, MessageType

class WorkflowOrchestrationAgent(BaseAgent):
    """
    Agent for orchestrating complex multi-agent workflows.
    """

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="WorkflowOrchestrationAgent",
            description="Orchestrates complex workflows across multiple agents"
        )
        self.logger = logging.getLogger(f"WorkflowOrchestrationAgent-{self.agent_id}")
        self.active_workflows = {}
        self.workflow_templates = {}
        self.agent_registry = {}

    async def initialize(self):
        """Initialize orchestration agent and workflow templates."""
        self.logger.info("Initializing Workflow Orchestration Agent...")
        
        # Define workflow templates
        self._define_workflow_templates()
        
        self.logger.info(f"Workflow Orchestration Agent initialized with {len(self.workflow_templates)} templates")

    async def cleanup(self):
        """Cleanup orchestration resources."""
        self.logger.info("Cleaning up Workflow Orchestration Agent...")
        
        # Cancel active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self._cancel_workflow(workflow_id)

    async def handle_query(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle workflow orchestration requests."""
        request_type = message.content.get('request_type')
        
        if request_type == 'start_workflow':
            return await self._start_workflow(message)
        elif request_type == 'get_workflow_status':
            return await self._get_workflow_status(message)
        elif request_type == 'cancel_workflow':
            return await self._cancel_workflow(message.content.get('workflow_id'))
        elif request_type == 'register_agent':
            return await self._register_agent(message)
        else:
            return {"error": f"Unknown request type: {request_type}"}

    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent for use in workflows."""
        self.agent_registry[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")

    async def _start_workflow(self, message: AgentMessage) -> Dict[str, Any]:
        """Start a new workflow."""
        workflow_type = message.content.get('workflow_type')
        workflow_params = message.content.get('params', {})
        
        if workflow_type not in self.workflow_templates:
            return {"error": f"Unknown workflow type: {workflow_type}"}
        
        workflow_id = f"{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_workflows)}"
        
        try:
            # Start workflow execution
            workflow_task = asyncio.create_task(
                self._execute_workflow(workflow_id, workflow_type, workflow_params)
            )
            
            self.active_workflows[workflow_id] = {
                'type': workflow_type,
                'params': workflow_params,
                'status': 'running',
                'started_at': datetime.now(),
                'task': workflow_task,
                'steps_completed': [],
                'current_step': None,
                'results': {}
            }
            
            self.logger.info(f"Started workflow {workflow_id} of type {workflow_type}")
            
            return {
                'workflow_id': workflow_id,
                'status': 'started',
                'message': f"Workflow {workflow_type} started successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error starting workflow {workflow_type}: {e}")
            return {"error": f"Failed to start workflow: {str(e)}"}

    async def _execute_workflow(self, workflow_id: str, workflow_type: str, params: Dict[str, Any]):
        """Execute a workflow according to its template."""
        try:
            workflow_info = self.active_workflows[workflow_id]
            template = self.workflow_templates[workflow_type]
            
            self.logger.info(f"Executing workflow {workflow_id}: {workflow_type}")
            
            # Execute workflow steps
            for step_name, step_config in template['steps'].items():
                workflow_info['current_step'] = step_name
                self.logger.info(f"Workflow {workflow_id}: Executing step {step_name}")
                
                try:
                    step_result = await self._execute_workflow_step(
                        workflow_id, step_name, step_config, params
                    )
                    
                    workflow_info['steps_completed'].append(step_name)
                    workflow_info['results'][step_name] = step_result
                    
                    # Update params with step results for next steps
                    if 'output_mapping' in step_config:
                        for output_key, param_key in step_config['output_mapping'].items():
                            if output_key in step_result:
                                params[param_key] = step_result[output_key]
                    
                except Exception as step_error:
                    self.logger.error(f"Workflow {workflow_id} step {step_name} failed: {step_error}")
                    workflow_info['status'] = 'failed'
                    workflow_info['error'] = str(step_error)
                    workflow_info['failed_step'] = step_name
                    return
            
            # Mark workflow as completed
            workflow_info['status'] = 'completed'
            workflow_info['completed_at'] = datetime.now()
            workflow_info['current_step'] = None
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} execution failed: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]['status'] = 'failed'
                self.active_workflows[workflow_id]['error'] = str(e)

    async def _execute_workflow_step(self, workflow_id: str, step_name: str, step_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        agent_name = step_config['agent']
        action = step_config['action']
        step_params = step_config.get('params', {})
        
        # Merge workflow params with step params
        merged_params = {**params, **step_params}
        
        # Get agent instance
        if agent_name not in self.agent_registry:
            raise Exception(f"Agent {agent_name} not registered")
        
        agent = self.agent_registry[agent_name]
        
        # Create message for agent
        message = AgentMessage(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            message_type=MessageType.QUERY,
            content={
                'request_type': action,
                **merged_params
            },
            timestamp=datetime.now()
        )
        
        # Execute agent action
        result = await agent.handle_query(message)
        
        self.logger.info(f"Workflow {workflow_id} step {step_name}: Agent {agent_name} completed action {action}")
        
        return result

    async def _get_workflow_status(self, message: AgentMessage) -> Dict[str, Any]:
        """Get status of a workflow."""
        workflow_id = message.content.get('workflow_id')
        
        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow_info = self.active_workflows[workflow_id]
        
        return {
            'workflow_id': workflow_id,
            'type': workflow_info['type'],
            'status': workflow_info['status'],
            'started_at': workflow_info['started_at'].isoformat(),
            'current_step': workflow_info['current_step'],
            'steps_completed': workflow_info['steps_completed'],
            'total_steps': len(self.workflow_templates[workflow_info['type']]['steps']),
            'results': workflow_info.get('results', {}),
            'error': workflow_info.get('error')
        }

    async def _cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a running workflow."""
        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow_info = self.active_workflows[workflow_id]
        
        if workflow_info['status'] == 'running':
            workflow_info['task'].cancel()
            workflow_info['status'] = 'cancelled'
            workflow_info['cancelled_at'] = datetime.now()
            
            self.logger.info(f"Cancelled workflow {workflow_id}")
            
            return {"message": f"Workflow {workflow_id} cancelled successfully"}
        else:
            return {"message": f"Workflow {workflow_id} is not running (status: {workflow_info['status']})"}

    def _define_workflow_templates(self):
        """Define workflow templates for different use cases."""
        
        # Article Processing Workflow
        self.workflow_templates['article_processing'] = {
            'description': 'Process articles through content analysis and storage',
            'steps': {
                'content_analysis': {
                    'agent': 'content_analysis',
                    'action': 'analyze_content',
                    'params': {},
                    'output_mapping': {
                        'quality': 'content_quality',
                        'insights': 'content_insights'
                    }
                },
                'classification': {
                    'agent': 'llm_coordinator',
                    'action': 'classification',
                    'params': {},
                    'output_mapping': {
                        'classification': 'content_category'
                    }
                },
                'summarization': {
                    'agent': 'llm_coordinator',
                    'action': 'summarization',
                    'params': {'max_length': 300},
                    'output_mapping': {
                        'summary': 'content_summary'
                    }
                }
            }
        }
        
        # Enhanced Search Workflow
        self.workflow_templates['enhanced_search'] = {
            'description': 'Enhanced search with query optimization and RAG',
            'steps': {
                'query_enhancement': {
                    'agent': 'llm_coordinator',
                    'action': 'search_query_enhancement',
                    'params': {},
                    'output_mapping': {
                        'enhanced_query': 'optimized_query'
                    }
                },
                'rag_search': {
                    'agent': 'search',
                    'action': 'search_and_analyze',
                    'params': {},
                    'output_mapping': {
                        'results': 'search_results',
                        'total_found': 'total_results'
                    }
                }
            }
        }
        
        # Content Enhancement Workflow
        self.workflow_templates['content_enhancement'] = {
            'description': 'Comprehensive content analysis and enhancement',
            'steps': {
                'quality_analysis': {
                    'agent': 'llm_coordinator',
                    'action': 'analyze_content',
                    'params': {'analysis_type': 'quality'},
                    'output_mapping': {
                        'analysis_result': 'quality_analysis'
                    }
                },
                'relevance_analysis': {
                    'agent': 'llm_coordinator',
                    'action': 'analyze_content',
                    'params': {'analysis_type': 'relevance'},
                    'output_mapping': {
                        'analysis_result': 'relevance_analysis'
                    }
                },
                'content_classification': {
                    'agent': 'llm_coordinator',
                    'action': 'classification',
                    'params': {},
                    'output_mapping': {
                        'classification': 'final_category'
                    }
                },
                'content_summarization': {
                    'agent': 'llm_coordinator',
                    'action': 'summarization',
                    'params': {'max_length': 500},
                    'output_mapping': {
                        'summary': 'enhanced_summary'
                    }
                }
            }
        }

    async def _register_agent(self, message: AgentMessage) -> Dict[str, Any]:
        """Register an agent via message."""
        agent_name = message.content.get('agent_name')
        agent_id = message.content.get('agent_id')
        
        # This would typically involve looking up the agent by ID
        # For now, return a placeholder response
        return {
            'message': f"Agent registration request received for {agent_name}",
            'status': 'pending'
        }
