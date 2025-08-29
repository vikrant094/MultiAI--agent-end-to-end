


# MultiAI--agent demonstration 

There are 3 different tasks, indivdually owned by agent called Researcher, Analyst, Writer. There is a supervisor managing this team of agents.

  1. Researcher - Gathers information and data
  2. Analyst - Analyzes data and provides insights
  3. Writer - Creates reports and summaries

Conditions : 
1. Researcher will do the research , then it will handles this information to supervisor .
2. Supervisor will handle it to analyst to do identify the trends and patterns. Analyst again send back analysis to supervisor.
3. Supervisor then take decision to handle it to writer , if he thinks analysis has been complete.
4. Then, Wrties wrties the complete report.

Input prompt is :
  I wants to build text recognition pipeline to enhance the capbility of inhouse invoice palidation pipeline.

Output summary :
![output report](./1.%20MultiAI-Agent/output/output.png?raw=true "Title")





Langsmith Studio has been used for debugging.


![input query](./1.%20MultiAI-Agent/output/input.png?raw=true "Title")
![langsmith snapshot](./1.%20MultiAI-Agent/output/langsmith_studio.png?raw=true "Title")


