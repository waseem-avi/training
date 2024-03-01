
def are_anagrams(str1, str2):

    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()

    return sorted(str1) == sorted(str2)

input_str1 = "listen"
input_str2 = "silent"
result = are_anagrams(input_str1, input_str2)
print("Are Anagrams:", result)