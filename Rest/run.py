from eve import Eve
import platform
import json
import socket
import getpass

settings = {'MONGO_HOST': 'localhost', 'MONGO_PORT': 27017, 'DOMAIN': {}}

app = Eve()


@app.route('/processor')
def details(settings=settings):
    data = {
        'processor': platform.processor(),
        'system': platform.system(),
        'architecture': platform.architecture()[0]
    }
    return json.dumps(data)

@app.route('/user')
def user():
    data = {
        'username': getpass.getuser(),
        'hostname': socket.gethostname(),
    }
    return json.dumps(data)

@app.route('/ram')
def ram():
    memory = Virtual_memory()
    data = {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'free':  memory.free,
        'cached': memory.cached,
        'shared': memory.shared,
    }
    
    return json.dumps(data)


if __name__ == '__main__':
    app.run()
