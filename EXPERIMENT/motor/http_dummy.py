'''
RUN BASED ON RANDOM
'''
# from flask import Flask
# import random

# app = Flask(__name__)

# @app.route('/')
# def command():
#     # value = 1
#     # print(f"Sent command: {value}")
#     # return str(value)
#     # for i in range(0,8):
#     #     print(f"Sent command: {i}")
#     #     return str(i)
#     value = random.randint(0, 7)
#     print(f"Sent command: {value}")
#     return str(value)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


'''
RUN BASED ON USER INPUT
'''
from flask import Flask
import threading

app = Flask(__name__)
current_command = "0"  # default

@app.route('/')
def command():
    print(f"ESP32 requested command → {current_command}")
    return current_command

def input_thread():
    global current_command
    while True:
        user_input = input("Enter command (0–7): ")
        if user_input in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            current_command = user_input
            print(f"Command updated to: {current_command}")
        else:
            print("⚠️ Invalid input. Please enter 0–7")

if __name__ == '__main__':
    # run input listener in parallel so Flask keeps serving requests
    threading.Thread(target=input_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
