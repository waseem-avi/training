# Q8: What is the name of highest paid person (including benefits)?
# A:
from data_frame import df
ans = df["TotalPayBenefits"].max()
print(ans)
