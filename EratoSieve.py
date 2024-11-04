while True:
 try:
    n = int(input("Please, enter any integer:"))
 except ValueError:
     print("This is not an integer.")
 else:
     break
a = list(range(0, n + 1))
i = 2
j = i**2
while i**2 <= n:
    for l in range(len(a)):
        while j <= n:
               a[j] = 0
               j = j + i
    i = i + 1
    j = i**2
print("Here are all the prime numbers before your integer:")
for l in range(2, len(a)):
    if a[l] != 0:
      print(a[l])