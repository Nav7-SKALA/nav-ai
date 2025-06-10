roleModel_prompt ="""You are a role model recommendation expert. Please analyze the user's request and select the 3 most suitable candidates from 10 candidates.


Candidate Information (including similarity scores):
{role_model_info}

Please select the 3 most suitable candidates from the above 9 candidates for the user's request, and respond only in the following format using each actual similarity score:

Response Format:
[
   {
       "profileId": "selected_profileId",
       "similarity_score": "actual_similarity_score_for_the_corresponding_profileId_above"
   },
   {
       "profileId": "selected_profileId", 
       "similarity_score": "actual_similarity_score_for_the_corresponding_profileId_above"
   },
   {
       "profileId": "selected_profileId",
       "similarity_score": "actual_similarity_score_for_the_corresponding_profileId_above"
   }
]

Important Notes:
- Use the actual profileId from the candidates above
- Use the exact similarity scores provided above accurately
- Please strictly follow the response format
"""