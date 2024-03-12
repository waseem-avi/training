# Q7: How much does JOSEPH DRISCOLL make (including benefits)?
# A: 24    270324.91
from data_frame import df
ans = df[df["EmployeeName"]=="JOSEPH DRISCOLL"]["TotalPayBenefits"]
print(ans)