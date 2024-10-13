from num2words import num2words
from keyword import iskeyword
a = input("Введите число от 0 до 99: ")
if str.isdigit(a):
    if int(a) > 99:
        print("Неправильный ввод")
    else:
           print(num2words(a, lang = 'ru'))
else:
    print("Неправильный ввод")