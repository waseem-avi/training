# Q6:What is the job title of JOSEPH DRISCOLL ? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll). 
# A: 24    CAPTAIN, FIRE SUPPRESSION
from data_frame import df
j_title = df [ df["EmployeeName"] == "JOSEPH DRISCOLL" ] ["JobTitle"]
print(j_title)