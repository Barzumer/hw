from math import sqrt, cos, radians
a = int(input("Please enter a length of the first side: "))
b = int(input("Now, enter a length of the second side: "))
c = int(input("Last but not least; the degree value of an angle between the sides: "))
if 0<c<180:
	c = radians(c)
	print(f"The length of the second side is: {sqrt((a ** 2) + (b ** 2) - (2 * (a * b) * (cos(c))))}.")
else:
	print("Invalid angle.")
