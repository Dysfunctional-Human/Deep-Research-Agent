# Deep Research Agent

A modular, graph-based research workflow built using **LangGraph**.  
The Deep Research Agent automatically:

- Generates a team of domain analysts  
- Conducts retrieval-augmented interviews (Web + Wikipedia)  
- Writes structured sections per analyst  
- Synthesizes everything into a polished **markdown research report**

This README is strictly based on the source files:
`Researcher.py`, `AnalystsGraph.py`, and `InterviewGraph.py`.

---
## 📁 Project Structure

```
Deep-Research-Agent/
├── .gitignore
├── LICENSE
├── README.md
├── Researcher.py
├── docker-compose.yml
├── langgraph.json
├── requirements.txt
├── sampleOutput.md
├── testing.ipynb
│
├── Prompts/
│   ├── AnalystInstructions.py
│   ├── InterviewInstructions.py
│   └── ResearchInstructions.py
│
└── SubGraphs/
    ├── AnalystsGraph.py
    └── InterviewGraph.py

```

---

## 🔧 Installation

### 1) Clone the repository
```
git clone https://github.com/Dysfunctional-Human/Deep-Research-Agent.git
cd Deep-Research-Agent
```
### 2) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3) Install dependencies
```
pip install -r requirements.txt
```

### 4) 🔑 Environment Variables
```
export GOOGLE_API_KEY="your_google_key"
export TAVILY_API_KEY="your_tavily_key"
```

## 🚀 Usage
```
from Researcher import DeepReasearchAgent

graph = DeepReasearchAgent.build_graph()

result = graph.run({
    "topic": "Impact of long-duration energy storage",
    "max_analysts": 3
})

print(result["final_report"])
```

### 📥 Inputs
| Key | Type | Description |
|----------|----------|----------|
| **topic**  | string  | Research topic  |
| **max_analysts**  | int  | Number of analysts to generate  |
| **messages**  | list  | (optional) message buffer  |
| **human_analyst_feedback** | string  | (optional) analyst revision input  |
		

### 📤 Outputs

**final_report** — full markdown report (intro, sections, conclusion, sources)

## 🧠 Internal Architecture
1. Analyst Generation — AnalystsGraph.py
    
    Models:

        Analyst

        AnalystTeam

    Produces analyst personas with:

        name

        role

        affiliation

        description

        Output: analysts

2. Interview Workflow — InterviewGraph.py
    
    Steps:

        generate_question

        search_web

        search_wikipedia

        generate_answer

        save_interview

        write_section

        Stops automatically on:

        max turns

        natural-ending phrases (“Thank you…”)

        Output: sections

3. Report Assembly — Researcher.py
        write_report

        write_introduction

        write_conclusion

        finalize_report

        Output: final_report

## 📝 Notes
1. Retrieval failures do not break execution

2. Structured outputs ensure consistent analysts

3. Final output is always clean markdown


