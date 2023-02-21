times = input("Enter times: ")
packet_time = float(input("Time in this packet: "))
total_time = float(input("Total time: "))

time1_in_minutes = int(times[0:2])*60 + int(times[3:5])
time2_in_minutes = int(times[6:8])*60 + int(times[9:])
elapsed_time = (time2_in_minutes - time1_in_minutes)/60.0

new_packet_time = round(packet_time + elapsed_time, 2)
new_total_time = round(total_time + elapsed_time, 2)

print(f"New Packet Time: {new_packet_time}\nNew Total: {new_total_time}\nAdded Time: {round(elapsed_time,2)}")
