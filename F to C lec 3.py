while True:
 try: 
    a = int(input("Please, enter the temperature in Fahrenheit:"))
 except ValueError:
    print("Please, enter a proper value")
 else:
    break
b = (a - 32) * 5/9
print("Your temperature is:",b,"°C")