import os

SAVE_PATH = "C:/Users/potot/Desktop/code/Research/Gymnasium/Saved Models/CarRacing256/"

def save():
    dir_list = os.listdir(SAVE_PATH)
    print(dir_list)
    if len(dir_list) >= 4:
        for folder in dir_list:
            filepath = os.path.join(SAVE_PATH,folder)
            if str(folder) == "1":
                os.rmdir(os.path.join(SAVE_PATH,folder))
                continue
            os.rename(filepath,f"{SAVE_PATH}{int(folder)-1}")
    return

save()