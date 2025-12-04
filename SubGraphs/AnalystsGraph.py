from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from Prompts.AnalystInstructions import analyst_instructions
import sys
import asyncio
from typing import Literal

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst"
    )
    name: str = Field(
        description="Name of the analyst"
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic"
    )
    description: str = Field(
        description="Description of the analyst focus, converns and motives"
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"
    
class AnalystTeam(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations"
    )
    
class GenerateAnalystsState(TypedDict, total=False):
    """State for analyst generation"""
    messages: Annotated[list[AnyMessage], add_messages]     # User and bot conversation
    topic: str  # Research topic
    max_analysts: int   # Number of analysts
    human_analyst_feedback: str     # Human feedback
    analysts: List[Analyst]  # Team of analysts
    
def create_analysts(state: GenerateAnalystsState) -> dict[str, any]:
    """Create the team of analysts

    Args:
        state (GenerateAnalystsState): Subgraph state for analyst creation

    Returns:
        dict[str, any]: Generated analysts
    """
    
    topic = state['topic']
    # print('hi-------')
    # print(state)
    max_analysts = state['max_analysts']
    try:
        human_analyst_feedback = state['human_analyst_feedback']
        state['human_analyst_feedback'] = None
    except:
        human_analyst_feedback = None
    
    gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = gemini.with_structured_output(AnalystTeam)
    
    system_message = analyst_instructions.format(topic=topic,
                                                human_analyst_feedback=human_analyst_feedback,
                                                max_analysts=max_analysts)

    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts")])
    
    return {"analysts": analysts.analysts, "messages": [topic]}

def human_feedback(state: GenerateAnalystsState):
    """Dummy node for human-in-the-loop interruption

    Args:
        state (GenerateAnalystsState): Subgraph state for analyst creation
    """
    pass
    
class AnalystGraph:
    def __init__(self):
        super(AnalystGraph, self).__init__()
        self.gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.instructions = analyst_instructions
    
    def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", END]:       #type: ignore
        """Return the next node to route to

        Args:
            state (GenerateAnalystsState): Subgraph state for analyst creation

        Returns:
            Literal["create_analysts", END]: Next node to execute
        """
        try:
            human_feedback = state['human_analyst_feedback']
        except:
            human_feedback = None
        
        if human_feedback and human_feedback != 'continue':
            state['messages'].append(human_feedback)
            return "create_analysts"

        return END
    
    def build_graph(self):
        """Build the Analyst creation graph

        Args:
            state (GenerateAnalystsState): Subgraph state for analyst creation
        
        Returns:
            graph: The analyst creation graph
        """
        builder = StateGraph(GenerateAnalystsState)
        builder.add_node("create_analysts", create_analysts)
        builder.add_node("human_feedback", human_feedback)
        
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.should_continue, ["create_analysts", END])
        
        memory = MemorySaver()
        graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
        
        return graph