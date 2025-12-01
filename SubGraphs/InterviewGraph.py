import operator
from typing import Annotated, List, Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from SubGraphs.AnalystsGraph import Analyst
from pydantic import BaseModel, Field
from Prompts.InterviewInstructions import question_instructions, search_instructions, answer_instructions, section_writer_instructions
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import sys
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import get_buffer_string
from langgraph.checkpoint.memory import MemorySaver

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class InterviewState(MessagesState):
    max_num_turns: int  # Max number of interview turns
    context: Annotated[List, operator.add]  # Source docs
    analyst: Analyst    # The analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API
    
class SearchQuery(BaseModel):
    search_query: str = Field(None , description="Search query for the retrieval")

def route_messages(state: InterviewState, name: str = "expert") -> Literal['save_interview', 'ask_question']:
    """Routes between interview and section writing

    Args:
        state (InterviewState): subgraph state for the interview

    Returns:
        dict[str, str]: interview transcript
    """
    # Get messages
    messages = state['messages']
    max_num_turns = state['max_num_turns']
    
    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    
    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'
    
    # This router is run after each question-answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return 'ask_question'
    
class InterviewAgent():
    def __init__(self):
        super(InterviewAgent, self).__init__()
        
        self.gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.question_instructions = question_instructions
        self.search_instructions = search_instructions
        self.answer_instructions = answer_instructions
        self.section_writer_instructions = section_writer_instructions
        
    def generate_question(self, state: InterviewState):
        """Node for an analyst to generate a question

        Args:
            state (InterviewState): Subgraph state for the interview

        Returns:
            dict[str, list[AIMessage]]: Interview transcript
        """
        # Get state
        analyst = state['analyst']
        messages = state['messages']
        
        # Generate question
        system_message = self.question_instructions.format(goals=analyst.persona)
        question = self.gemini.invoke([SystemMessage(content=system_message)] + messages)
        
        # Write messages to state
        return {"messages": [question]}
    
    def search_web(self, state: InterviewState):
        """Retrieve documents from the web

        Args:
            state (InterviewState): subgraph state for the interview

        Returns:
            dict[str, list[str]]: List of source docs
        """
        # Search query
        structured_gemini = self.gemini.with_structured_output(SearchQuery)
        search_query = structured_gemini.invoke([self.search_instructions]+state['messages'])
        
        try:
            # Search
            tavilly_search = TavilySearchResults(max_results=3)
            search_docs = tavilly_search.invoke(search_query.search_query)
            
            # Format
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    for doc in search_docs
                ]
            )
            
            return {"context": [formatted_search_docs]}
        except Exception as e:
            return {"context": [""]}
    
    def search_wikipedia(self, state: InterviewState):
        """Retrieve documents from wikipedia

        Args:
            state (InterviewState): subgraph state for the interview

        Returns:
            dict[str, list[str]]: List of source docs
        """
        
        # Search query
        structured_gemini = self.gemini.with_structured_output(SearchQuery)
        search_query = structured_gemini.invoke([search_instructions] + state['messages'])
        
        try:
            # Search
            search_docs = WikipediaLoader(query=search_query.search_query,
                                          load_max_docs=2).load()
                
            # Format
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in search_docs
                ]
            )

            return {"context": [formatted_search_docs]}  
        except Exception as e:
            return {"context": [""]}
        
    def generate_answer(self, state: InterviewState):
        """Node for an expert to answer an analyst's question

        Args:
            state (InterviewState): subgraph state for the interview

        Returns:
            dict[str, list[str]]: answer to the analyts's question
        """
        # Get state
        analyst = state['analyst']
        messages = state['messages']
        context = state['context']
        
        # Answer question
        system_message = self.answer_instructions.format(goals=analyst.persona, context=context)
        answer = self.gemini.invoke([SystemMessage(content=system_message)] + messages)
        
        # Name the message as coming from the expert
        answer.name = "expert"
        
        # Append it to the state
        return {"messages": [answer]}
    
    def save_interview(self, state: InterviewState):
        """Saves the interview transcript as a string

        Args:
            state (InterviewState): subgraph state for the interview

        Returns:
            dict[str, str]: interview transcript
        """
        # Get messages
        messages = state['messages']
        
        # Convert interview to a string
        interview = get_buffer_string(messages=messages)
        
        # Save to interviews key
        return {"interview": interview}
    
    def write_section(self, state: InterviewState):
        """Node to paraphrase the insights derived from the interview into a formal document

        Args:
            state (InterviewState): subgraph state for the Interview

        Returns:
            dict[str, list[str]]: The report content
        """
        # Get state
        context = state['context']
        analyst = state['analyst']
        
        # Write the section using either the gathered docs from the interview (context) or the interview itself (interview)
        system_message = self.section_writer_instructions.format(focus=analyst.description)
        section = self.gemini.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}")])
        
        # Append it to the state
        return {'sections': [section.content]}
    
    def build_graph(self):
        """Building the Interview sub-graph

        Returns:
            interviewGraph: Compiled graph for conducting interviews
        """
        # Creating nodes
        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", self.generate_question)
        interview_builder.add_node("search_web", self.search_web)
        interview_builder.add_node("search_wikipedia", self.search_wikipedia)
        interview_builder.add_node("answer_question", self.generate_answer)
        interview_builder.add_node("save_interview", self.save_interview)
        interview_builder.add_node("write_section", self.write_section)
        
        # Create edges and flow
        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)
        
        memory = MemorySaver()
        interviewGraph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")
        
        return interviewGraph