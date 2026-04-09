import socket

target = "127.0.0.1"
port = 80

while True:
    try:
        s = socket.socket()
        s.connect((target, port))
        s.close()
    except:
        pass
