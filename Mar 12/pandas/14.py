# Q14:How many people have the word Chief in their job title? (This is pretty tricky)
# A:627
from data_frame import df
def chief(title):
    if 'chief' in title.lower():
        return True
    else:
        return False
print(sum(df['JobTitle'].apply(chief)))