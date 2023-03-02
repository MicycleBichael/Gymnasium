packet_time, total_time = 0.0, 0.0
with open('Gymnasium/hours.txt', 'r') as file:
    data = file.read().split('\n')
    packet_time = float(data[0][14:])
    total_time = float(data[1][13:])
while(True):
    print(f"Packet time: {packet_time} | Total time: {total_time}")
    option = input("> (E)nter times\n> E(d)it times\n> (N)ew Packet\n> (Q)uit\n>> ").lower()
    if option == "e":
        times = input("Enter time: ")
        elapsed_time = 0.0
        try:
            time1_in_minutes = int(times[0:2])*60 + int(times[3:5])
            time2_in_minutes = int(times[6:8])*60 + int(times[9:])
            elapsed_time = (time2_in_minutes - time1_in_minutes)/60.0
        except:
            elapsed_time = float(times)
        packet_time = round(packet_time + elapsed_time, 2)
        total_time = round(total_time + elapsed_time, 2)
        print(f"New Packet Time: {packet_time}\nNew Total: {total_time}\nAdded Time: {round(elapsed_time,2)}")
        with open('Gymnasium/hours.txt', 'w') as file:
            L = [f"Packet hours: {packet_time}\n",f"Total hours: {total_time}"]
            file.writelines(L)
    elif option == "d":
        packet_time = float(input("New packet time: "))
        total_time = float(input("New total time: "))
        with open('Gymnasium/hours.txt', 'w') as file:
            L = [f"Packet hours: {packet_time}\n",f"Total hours: {total_time}"]
            file.writelines(L)
    elif option == "n":
        packet_time = 0.0
        with open('Gymnasium/hours.txt', 'w') as file:
            L = [f"Packet hours: {packet_time}\n",f"Total hours: {total_time}"]
            file.writelines(L)
    elif option == "q":
        break
