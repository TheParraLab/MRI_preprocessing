from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import subprocess
import os
import threading
import re
import logging
import datetime

DATA_DIR = '/FL_system/data/'
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Regular expression to match ANSI escape codes
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

#############################################
### Helper functions
def get_current_time():
    # Get current time in ISO format
    return datetime.datetime.now().isoformat()

def get_container_name(action):
    # Map actions to container names
    # This is used to fetch logs for the specific container within the emit_command_output function
    container_map = {
        'startSuperLink': 'superlink'
    }
    return container_map.get(action, '')

def extract_node_id(log_line):
    # Extract node ID from log line
    # TODO: update function to map node ID to human-readable name
    print(log_line)
    # Match node creation
    match = re.search(r'INFO\s*:\s*\[Fleet.CreateNode\]\s*Created\s*node_id=(-?\d+)', log_line)
    if match:
        return match.group(1), 'active'
    # Match node deletion
    match = re.search(r'INFO\s*:\s*\[Fleet.DeleteNode\]\s*Delete\s*node_id=(-?\d+)', log_line)
    if match:
        return match.group(1), 'inactive'
    return None, None

### Function to execute a command and emit the output back to the client
# This function is called in a separate thread to prevent blocking the main thread
# Depenging on the function called, the terminal output is emitted back to the client, and 
def emit_command_output(command, action):
    # Execute command and emit output back to client
    print(f'Executing command: {command}')    
    try:
        # Execute the provided command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True,cwd='/FL_system')

        ## Monitor outputs depending on the supplied action
        
        #if action in ['startClient', 'stopClient', 'processData']:
            # If the action is to start/stop the client, send command_status to update the client status indicator
            # If the action is to process data, send command_status to update the data processing status indicators for each step
        ProcessCompletion = False
        for line in iter(process.stdout.readline, ''):
            socketio.emit('command_output', {'data': line})
            if (not ProcessCompletion) and "fl_client" in line:
                ProcessCompletion = True
                if action == 'startClient':
                    socketio.emit('command_status', {'status': 'active'})
                elif action == 'stopClient':
                    socketio.emit('command_status', {'status': 'inactive'})
            if line == "01 Completed\n":
                socketio.emit('command_status', {'status': 'completed', 'step': '01'})
            elif line == "02 Completed\n":
                socketio.emit('command_status', {'status': 'completed', 'step': '02'})
            elif line == "03 Completed\n":
                socketio.emit('command_status', {'status': 'completed', 'step': '03'})
            elif line == "04 Completed\n":
                socketio.emit('command_status', {'status': 'completed', 'step': '04'})
            elif line == "05 Completed\n":
                socketio.emit('command_status', {'status': 'completed', 'step': '05'})
        process.stdout.close()
        process.wait()

        # If the action is to start the super link, fetch logs from the container and emit them to the webpage
        #if action in ['startSuperLink']:
        containter_name = get_container_name(action) # Get the container name
        current_time = get_current_time() # Get the current time
        print('Fetching logs since:', current_time) # Ensure logs are fetched from the current time forward
        log_process = subprocess.Popen(f'docker logs --since {current_time} -f {containter_name}', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
        for log_line in iter(log_process.stdout.readline, ''):
            # Remove ANSI escape codes
            clean_log_line = ansi_escape.sub('', log_line)
            # Emit log line to client
            socketio.emit('command_output', {'data': clean_log_line, 'action': action})
            # check if the log includes a node reference
            node_id, status = extract_node_id(clean_log_line)
            print(f'Node ID: {node_id}')
            if node_id and status=='active':
                # if the node is created, emit the node_active event
                print('node started:', node_id)
                socketio.emit('node_active', {'node_id': node_id})
            elif node_id and status=='inactive':
                # if the node is deleted, emit the node_inactive event
                print('Node stopped:', node_id)
                socketio.emit('node_inactive', {'node_id': node_id})
        log_process.stdout.close()
        log_process.wait()
    except Exception as e:
        socketio.emit('command_output', {'data': f'Error: {str(e)}', 'action':action})
### End of helper functions
#############################################

#############################################
### Routes
@app.route('/')
# Serves the index.html page
# The dataPath variable is passed to the template for display to the user
def home():
    data_directory_path = os.getenv('DATA_DIRECTORY_PATH', 'Default Path')
    return render_template('index.html', dataPath=data_directory_path)

### Custom routes for each page
@app.route('/client.html')
# Fills in the containers into the template page
def client():
    data_directory_path = os.getenv('DATA_DIRECTORY_PATH', 'Default Path')
    return render_template('client.html', dataPath=data_directory_path)
@app.route('/server.html')
# Fills in the containers into the template page
def server():
    data_directory_path = os.getenv('DATA_DIRECTORY_PATH', 'Default Path')
    return render_template('server.html', dataPath=data_directory_path)
### End of custom routes for pages

@app.route('/gpu-status')
# Checks if the GPU is available
def gpu_status():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # If the command was successful, and the output contains GPU info
        if "NVIDIA-SMI" in result.stdout:
            return jsonify({'status': 'available'})
        else:
            return jsonify({'status': 'unavailable'})
    except subprocess.CalledProcessError:
        # nvidia-smi command failed
        return jsonify({'status': 'unavailable'})
    
@app.route('/client-status')
# Checks if the client container is running
def client_status():
    try:
        # Command to list all containers and filter by name 'fl_client'
        command = ["docker", "ps", "-a", "--filter", "name=fl_client", "--format", "{{.Names}}"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # If the command was successful, and the output contains 'fl_client'
        if "fl_client" in result.stdout:
            # Further check if the container is running
            command = ["docker", "inspect", "-f", "{{.State.Running}}", "fl_client"]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if result.stdout.strip() == "true":
                return jsonify({'status': 'active'})
            else:
                return jsonify({'status': 'inactive'})
        else:
            return jsonify({'status': 'unavailable'})
    except subprocess.CalledProcessError:
        # nvidia-smi command failed
        return jsonify({'status': 'unavailable'})

### Preprocessing status routes
@app.route('/scan-raw')
def scan_data():
    # Scan the raw data directory and return the list of files
    logger.info('Scanning raw data directory...')
    try:
        files = os.listdir(f'{DATA_DIR}raw')
        n = len(files)
        if n == 1:
            files = os.listdir(f'{DATA_DIR}raw/' + files[0])
            n = len(files)
        return jsonify({'message': 'success', 'data': files, 'count': n})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/details-extracted')
def details_extracted():
    # Check if the data table has been extracted
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'Data_table.csv' in files:
            return jsonify({'message': 'success'})
        return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/details-parsed')
def details_parsed():
    # Check if the data table has been parsed
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'Data_table_timing.csv' in files:
            return jsonify({'message': 'success'})
        return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/nifti-converted')
def nifti_converted():
    # Check if the data has been converted to NIfTI format
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'nifti' in files:
            files2 = os.listdir(f'{DATA_DIR}nifti/')
            n = len(files2)
            return jsonify({'message': 'success', 'data': files2, 'count': n})
        else: return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/RAS-converted')
def RAS_converted():
    # Check if the data has been converted to RAS format
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'RAS' in files:
            files2 = os.listdir(f'{DATA_DIR}RAS/')
            n = len(files2)
            return jsonify({'message': 'success', 'data': files2, 'count': n})
        return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/coregistered')
def coregistered():
    # Check if the data has been coregistered
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'coreg' in files:
            files2 = os.listdir(f'{DATA_DIR}coreg/')
            n = len(files2)
            return jsonify({'message': 'success', 'data': files2, 'count': n})
        return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
@app.route('/inputs-generated')
def input_generated():
    # Check if the input data is ready
    try:
        files = os.listdir(f'{DATA_DIR}')
        if 'inputs' in files:
            files2 = os.listdir(f'{DATA_DIR}inputs/')
            n = len(files2)
            return jsonify({'message': 'success', 'data': files2, 'count': n})
        return jsonify({'message': 'failure'})
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
    
### End of preprocessing status routes

#############################################
### SocketIO events
# This function is called when a user asks the serve to run a command: i.e., start the client, stop the client, etc.
# The command is executed in a separate thread to prevent blocking the main thread
# The desired action is passed to the emit_command_output function
@socketio.on('start_command')
def handle_start_command(json):
    '''
     Defines the command to be executed based on the action received from the client
    ############################################################
     !!!For security reasons, only predefined actions are allowed!!!
    ############################################################
     The command is executed in a separate thread to prevent blocking the main thread
    '''
    # Extract action from JSON
    action = json['action']
    logger.info(f'Action received: {action}')

    if action == 'startClient':
        # Start the client
        logger.info('Attempting to start client...')
        command = 'bash start_client.sh'
    elif action == 'stopClient':
        # Stop the client
        print('Attempting to stop client...')
        command = 'docker compose -f ./sample-project/docker-compose-client.yml down'
    elif action == 'processData':
        # Start the data processing pipeline
        print('Attempting to process data...')
        command = 'bash /FL_system/code/preprocessing/00_preprocess.sh'
    elif action == 'startSuperNode':
        # Start the super node - DEPRECATED
        print('Attempting to start super node...')
        command = 'docker compose -f ./client_system/docker-compose-supernode.yml up'
    elif action == 'startSuperLink':
        # Start the super link
        # This initializes the FL server, and the supernodes will connect to it
        print('Attempting to start super link...')
        command = 'docker compose -f ./sample-project/docker-compose-server.yml up -d'
    elif action == 'startQuickstart':
        # Start the quickstart scenario
        # Launches the FL server, and 2 complete clients with supernode and clientapp
        print('Attempting to start quickstart...')
        command = 'bash start_quickstart.sh'
    threading.Thread(target=emit_command_output, args=([command], action)).start()
### End of SocketIO events
#############################################



#############################################
### Main
# Start the Flask app
if __name__ == '__main__':
    logger.info('Starting Flask app... ')
    app.run(debug=True, host='0.0.0.0')
#############################################