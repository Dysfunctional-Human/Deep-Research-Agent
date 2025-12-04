import operator
from typing import List, Annotated
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from SubGraphs.AnalystsGraph import AnalystGraph, Analyst, create_analysts, human_feedback
from SubGraphs.InterviewGraph import InterviewAgent
from Prompts.ResearchInstructions import report_writer_instructions, intro_conclusion_instructions

analystGraph = AnalystGraph()
Team = analystGraph.build_graph()
print('Analyst Creation Graph Active')

interviewGraph = InterviewAgent()
interviewSetup = interviewGraph.build_graph()
print('Interview Graph Active')

class ResearchGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]     # Conversation
    topic: str  # Research topic
    max_analysts: int   # Number of analysts
    human_analyst_feedback: str     # Human Feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str   # Introduction of the final report
    content: str    # Content of the final report
    conclusion: str # Conclusion of the final report
    final_report: str # Compiled final report

class InputResearchGraphState(TypedDict, total=False):
    topic: str  # The topic of research
    max_analysts: int   # Number of analysts
    messages: Annotated[list[AnyMessage], add_messages] # Conversation
    human_analyst_feedback: str     # Human Feedback

def initiate_all_interviews(state: ResearchGraphState):
    """Decides whether to redo analyst creation or start conducting interviews

    Args:
        state (ResearchGraphState): Graph state for the research graph

    Returns:
        list[Send] | Literal['create_analysts']: The next node to go to
    """
    try:
        human_analyst_feedback = state['human_analyst_feedback']
    except:
        human_analyst_feedback = None    # Check if human feedback is present
        
    if human_analyst_feedback:
        # Return to create analysts
        return "create_analysts"
    
    # Otherwise kickoff interviews in parall via the Send() API
    else:
        topic = state['topic']
        print('hi-------')
        print(state)
        return [Send("conduct_interview", {"analyst": analyst,
                                            "messages": [HumanMessage(
                                                content=f"So you said you were writing an article on {topic}?"
                                            )
                                                        ]}) for analyst in state['analysts']]
    
class ResearchAgent():
    def __init__(self):
        super(ResearchAgent, self).__init__()
        self.gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.report_instructions = report_writer_instructions
        self.intro_conclusion_instructions = intro_conclusion_instructions

    def write_report(self, state: ResearchGraphState):
        """Write the body of the final report

        Args:
            state (ResearchGraphState): Graph state for the research graph

        Returns:
            dict[str, str]: final body
        """
        # Full set of sections
        sections = state['sections']
        topic = state['topic']
        
        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        system_message = self.report_instructions.format(topic=topic, context=formatted_str_sections)    
        report = self.gemini.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
        return {"content": report.content}
    
    def write_introduction(self, state: ResearchGraphState):
        """Write the introduction of the final report

        Args:
            state (ResearchGraphState): Graph state for the research graph

        Returns:
            dict[str, str]: final introduction of the report
        """
        # Full set of sections
        sections = state['sections']
        topic = state['topic']
        
        # Concat all the sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a introduction
        instructions = self.intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        intro = self.gemini.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")])
        return {"introduction": intro.content}
    
    def write_conclusion(self, state: ResearchGraphState):
        """Write the conclusion of the final report

        Args:
            state (ResearchGraphState): Graph state for the research graph

        Returns:
            dict[str, str]: final conclusion of the report
        """
        # Full set of sections
        sections = state['sections']
        topic = state['topic']
        
        # Concat all the sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a introduction
        instructions = self.intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        conclusion = self.gemini.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")])
        return {"conclusion": conclusion.content}
    
    def finalize_report(self, state: ResearchGraphState):
        """Format the parts and content into a final formal research report

        Args:
            state (ResearchGraphState): Graph state for the research graph

        Returns:
            dict[str, str]: Final research report
        """
        # Save the full final report
        content = state["content"]
        if content.startswith('## Insights'):
            content=content.strip('## Insights')
        
        if '## Sources' in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None
            
        final_report = state['introduction'] + "\n\n---\n\n" + content + "\n\n---\n\n" + state['conclusion']
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        
        print(final_report)
        return {"final_report": final_report, "messages": [final_report]}
    
    def build_graph(self):
        """Build the final research agent
        
        Returns:
            researcher: Deep Research Agent
        """
        # Add Nodes
        builder = StateGraph(input_schema=InputResearchGraphState, state_schema=ResearchGraphState)
        builder.add_node("create_analysts", create_analysts)
        builder.add_node("human_feedback", human_feedback)
        builder.add_node("conduct_interview", interviewSetup)
        builder.add_node("write_report", self.write_report)
        builder.add_node("write_introduction",self.write_introduction)
        builder.add_node("write_conclusion",self.write_conclusion)
        builder.add_node("finalize_report",self.finalize_report)
        
        # Logic
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)
        
        # Compile
        memory = MemorySaver()
        researcher = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
        return researcher
    
DeepReasearchAgent = ResearchAgent()
graph = DeepReasearchAgent.build_graph()

if __name__ == "__main__":
    print("Graph compiled successfully!")
    print(f"Graph nodes: {graph.nodes}")