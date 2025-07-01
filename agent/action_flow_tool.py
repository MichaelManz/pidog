from langchain.tools import BaseTool
from action_flow import ActionFlow

class ActionFlowTool(BaseTool):
    name: str = "ActionFlowTool"
    description: str = None
    action_flow: ActionFlow = None
    action: str = None

    def __init__(self, action_flow: ActionFlow, action: str):
        super().__init__()
        self.action_flow = action_flow
        self.action = action
        self.description = f"Use this tool to make the dog execute the action {action}"

    def _run(self) -> int:
        self.action_flow.run(self.action)