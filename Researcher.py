from SubGraphs.AnalystsGraph import AnalystGraph
from SubGraphs.InterviewGraph import InterviewAgent

analystGraph = AnalystGraph()
Team = analystGraph.build_graph()
print('Analyst Creation Graph Active')

interviewGraph = InterviewAgent()
interviewSetup = interviewGraph.build_graph()
print('Interview Graph Active')