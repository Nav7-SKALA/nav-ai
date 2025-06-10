careerSummary_prompt='''**The career data contains information about an employee's career history up to now.**
Please analyze the following career data and summarize it in the same format as the **example output** below, referring to the example:

**Career Data:** {profile_data}

**Example Output Format:**
**ooo is a X-year backend development expert.**
* 🔹 Total Projects: 12 projects
* 🔹 Certifications: AWS Solutions Architect, OCP, Information Processing Engineer (Total 3)
* 🔹 Core Tech Stack: Python, Spring Boot, Docker, MySQL, AWS

**Major Achievements**
1. Company A order management system refactoring → 30% response speed improvement
2. Led Company B infrastructure automation project
3. Implemented Kubernetes-based deployment pipeline after obtaining OCP certification

**Recommended Insights**
* Consider learning C certification to supplement "cloud cost optimization" skills
* Propose strengthening "microservice architecture design" capabilities as next goal

**Writing Guidelines:**
* Display employee number with "님" (Korean honorific)
* Calculate total career based on highest year of experience
* Select 1 most important specialized field
* Count number of projects from data
* List only major tech stacks with duplicates removed
* Select 3 major achievements based on project scale or importance
* Recommend insights with simple suggestions for strengths and development directions
⚠️ All responses must be written in Korean.
'''