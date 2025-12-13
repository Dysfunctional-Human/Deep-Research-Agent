# Deep Research Agent

A modular, graph-based research workflow built using **LangGraph**.  
The Deep Research Agent automatically:

- Generates a team of domain analysts  
- Conducts retrieval-augmented interviews (Web + Wikipedia)  
- Writes structured sections per analyst  
- Synthesizes everything into a polished **markdown research report**


## 🔗 Graph Structure
<img width="598" height="507" alt="Screenshot 2025-12-02 000031" src="https://github.com/user-attachments/assets/fdfb34b8-7126-4952-9506-ac96d49e0fb4" />

#### Inside the conduct_interview sub-graph
<img width="677" height="535" alt="Screenshot 2025-12-01 235953" src="https://github.com/user-attachments/assets/5211f55b-d2c9-48c0-8a16-0f4f0199aa61" />

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

To run the file, just open the terminal in base directory and
```
langgraph build -t my-image
docker compose up
```
Then navigate to langsmith to open LangGraph Studio

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


