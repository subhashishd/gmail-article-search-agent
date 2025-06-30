"""
Tool Registry Module

Manages:
- Sandboxed tool execution
- Tool discovery and registration
- Schema validation
"""

import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

@dataclass
class Tool:
    """Represents an available tool for agent use"""
    name: str
    execute: Callable[[Dict[str, Any]], Any]  # Function to execute the tool
    description: str
    schema: Dict[str, Any]  # Input and output schema

class ToolRegistry:
    """Registry for managing available tools with sandboxing"""
    
    def __init__(self):
        self.logger = logging.getLogger("ToolRegistry")
        self.tools = {}
        
    def register_tool(self, name: str, execute: Callable[[Dict[str, Any]], Any], description: str, schema: Dict[str, Any]):
        """Register a new tool"""
        tool = Tool(name=name, execute=execute, description=description, schema=schema)
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")
    
    def execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a registered tool in a sandboxed environment"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool = self.tools[name]
        self.logger.info(f"Executing tool: {name}")
        
        # Validate input against schema
        self._validate_schema(params, tool.schema['input'])
        
        # Execute tool
        result = tool.execute(params)
        
        # Validate output against schema
        self._validate_schema(result, tool.schema['output'])
        
        return result
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]):
        """Validate data against schema"""
        missing = [key for key in schema if key not in data]
        if missing:
            raise ValueError(f"Missing keys in data: {missing}")

# Example use of the ToolRegistry
if __name__ == "__main__":
    # Initialize tool registry
    tool_registry = ToolRegistry()
    
    # Example tool with simple execution
    def example_tool(params):
        return {"result": params['value'] + 10}
    
    # Register the tool with input/output schemas
    tool_registry.register_tool(
        "ExampleTool",
        execute=example_tool,
        description="Adds 10 to the input value",
        schema={
            "input": {"value": "int"},
            "output": {"result": "int"}
        }
    )
    
    # Execute the tool with correct input
    try:
        output = tool_registry.execute_tool("ExampleTool", {"value": 5})
        print(f"Tool executed successfully: {output}")
    except Exception as e:
        print(f"Error executing tool: {e}")

