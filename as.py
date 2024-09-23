from math import sqrt, cos, degrees
a = input("Please enter a length of the first side: ")
b = input("Now, enter a length of the second side: ")
с = input("Last but not least; the angle between the sides: ")
print(sqrt((a ** 2) + (b ** 2) - (2 * (a * b) * degrees(cos(c)))))
input()