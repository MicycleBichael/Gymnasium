a = input("Enter times: ")
d = input("Total time: ")
c = round(((float(a[6:8])*60 + float(a[9:])) - (float(a[0:2])*60 + float(a[3:5])))/60, 2)
print(f"New Total: {float(d)+c}\nAdded Time: {c}")
