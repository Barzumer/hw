import random
a = (random.randrange(99))
b = int(input("Guess a number from 0 to 99:"))
while(a != b):
 if b < 0:
    print("Did I stutter? Pick a proper number!")
    b = int(input("Try again:"))
 elif b > 99:
    print("Did I stutter? Pick a proper number!")
    b = int(input("Try again:"))
 elif b > a:
    print("Wrong! My number is less than that")
    b = int(input("Try again:"))
 elif b < a:
    print("Wrong! My number is bigger than that")
    b = int(input("Try again:"))
      
print("Correct! GG")
input()