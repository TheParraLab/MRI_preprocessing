///////////////////////////////////////////////////////////////
// This is the primary script for the web interface
// It contains all the functions that interact with the server
// The script is divided into sections based on the functionality
// Each section contains functions that perform a specific task
///////////////////////////////////////////////////////////////
var socket = io();
////////////////////////// ONLOAD //////////////////////////
document.addEventListener('containersContentLoaded', function() {
    // Update statuses once containers have been loaded
    fetchGPUStatus();
    fetchClientStatus();
    scanData();
});

////////////////////////// FETCH API //////////////////////////
async function fetchGPUStatus() {
    // Fetch GPU status from the system
    try {
        const response = await fetch('/gpu-status');
        const data = await response.json();
        const statusElement = document.getElementById('gpuStatus');
        if(data.status === 'available') {
            statusElement.textContent = 'GPU Available';
            statusElement.classList.replace('inactive', 'active');
        } else {
            statusElement.textContent = 'GPU Unavailable';
        }
    } catch (error) {
        console.error('Error fetching GPU status:', error);
        document.getElementById('gpuStatus').textContent = 'Error fetching GPU status';
    }
}
async function fetchClientStatus() {
    // Fetch client status from the system
    try{
        const response = await fetch('/client-status');
        const data = await response.json();
        const statusElement = document.getElementById('clientStatus');
        if(data.status === 'active') {
            statusElement.textContent = 'Client running';
            statusElement.classList.replace('inactive', 'active');
            document.getElementById('startClient').textContent = 'Stop Client';
        } else {
            statusElement.textContent = 'Client inactive';
        }
    } catch (error) {
        console.error('Error fetching client status:', error);
        document.getElementById('clientStatus').textContent = 'Error fetching client status';
    }
}

////////////////////////// SCAN DATA ///////////////////
async function scanData() {
    // Scan the data directory for available samples
    // Also check how far the data has been processed
    try {
        let response = await fetch('/scan-raw');
        let data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('dataSize').textContent = `${data.count} available samples`;
            document.getElementById('RawPresent').checked=true;
        } else {
            document.getElementById('dataStatus').textContent = 'No data available';
            alert('Failed to scan data directory');
            //End function if no data is available
            return;
        }
        response = null;
        data = null;

        response = await fetch('/details-extracted');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('DetailsExtracted').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'preprocessing pending';
            //End function if preprocessing is pending
            return;
        }
        response = null;
        data = null;
        
        response = await fetch('/details-parsed');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('DetailsParsed').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'preprocessing pending';
            //End function if preprocessing is pending
            return;
        }
        response = null;
        data = null;

        response = await fetch('/nifti-converted');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('NiftiConversion').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'pending nifti conversion';
            //End function if nifti conversion is pending
            return;
        }
        response = null;
        data = null;

        response = await fetch('/RAS-converted');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('RasComplete').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'pending RAS conversion';
            //End function if RAS conversion is pending
            return;
        }
        response = null;
        data = null;

        response = await fetch('/coregistered');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('Aligned').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'pending coregistration';
            //End function if coregistration is pending
            return;
        }
        response = null;
        data = null;

        response = await fetch('/inputs-generated');
        data = await response.json();
        console.log(data);
        if (data.message === "success") {
            document.getElementById('InputsGen').checked=true;
        } else if (data.message === "failure") {
            document.getElementById('dataStatus').textContent = 'pending input generation';
            //End function if input generation is pending
            return;
        }
    
    } catch (error) {
        console.error('Error scanning data:', error);
        alert('Failed to scan data');
    }
}

////////////////////////// TERMINAL //////////////////////////
function scrollToBottom() {
    var terminalOutput = document.querySelector('.terminalOutput');
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
  }
////////////////////////// Button Interaction //////////////////////////
document.addEventListener('DOMContentLoaded', function() {
    document.body.addEventListener('click', function(event) {
        // All button interactions are handled in this block 
        // this ensures that the buttons will operate even for elements not initially loaded

        // Check if the clicked element has the ID 'clearTerminal'
        if (event.target.id === 'clearTerminal') {
            // Clear the 'terminalOutput' content
            document.getElementById('terminalOutput').innerHTML = '';

        // Check if the clicked element has the ID 'processData'
        } else if (event.target.id === 'processData') {
            console.log('Processing data...')
            try {
                console.log('Attempting to send command to server...')
                //alert('Button not piped');
                socket.emit('start_command', {action: 'processData'}); // Example command
        
            }
            catch (error) {
                console.error('Error processing data:', error);
                alert('Failed to process data');
            }

        // Check if the clicked element has the ID 'startClient'
        } else if (event.target.id === 'startClient') {
            console.log('Toggling client...')
            const isGPUAvailable = document.getElementById('gpuStatus').classList.contains('active');
            if (!isGPUAvailable) {
                console.log('GPU is not available. Please check the GPU status.')
                alert('GPU is not available. Please check the GPU status.');
                return;
            }
            const clientStatusElement = document.getElementById('clientStatus');
            const isClientActive = clientStatusElement.classList.contains('active');

            // Determine the appropriate action based on the client's current status
            const actionCommand = isClientActive ? 'stopClient' : 'startClient';
            const actionMethod = isClientActive ? 'Stopping' : 'Starting';

            try {
                clientStatusElement.textContent = `${actionMethod} client...`;
                console.log('Attempting to send command to server...')
                console.log(actionCommand)
                socket.emit('start_command', {action: actionCommand}); // Example command
            }
            catch (error) {
                console.error(`Error ${actionMethod.toLowerCase()} client:`, error);
                alert(`Failed to ${actionMethod.toLowerCase()} client`);
            }
        
        // TESTING: SUPERNODE docker container
        } else if (event.target.id === 'startSuperNode') {
            console.log('Starting SuperNode...')
            try {
                console.log('Attempting to send command to server...')
                socket.emit('start_command', {action: 'startSuperNode'}); // Example command
            }
            catch (error) {
                console.error('Error starting SuperNode:', error);
                alert('Failed to start SuperNode');
            }
        } else if (event.target.id =='startSuperLink'){
            console.log('Starting SuperLink...')
            try {
                console.log('Attempting to send command to server...')
                socket.emit('start_command', {action: 'startSuperLink'}); // Example command
            }
            catch (error) {
                console.error('Error starting SuperLink:', error);
                alert('Failed to start SuperLink');
            }
        } else if (event.target.id == 'startQuickstart'){
            console.log('Starting Quickstart...')
            try {
                console.log('Attempting to send command to server...')
                socket.emit('start_command', {action: 'startQuickstart'}); // Example command
            }
            catch (error) {
                console.error('Error starting Quickstart:', error);
                alert('Failed to start Quickstart');
            }
        }
    });
});
////////////////////////// SOCKET IO //////////////////////////
// All socket.io code is placed in this block
// This ensures that the code is executed only after the DOM is fully loaded
// socket.io is used to return continuous output from the server
// This output is then displayed in the terminalOutput div

document.addEventListener('containersContentLoaded', function () {
    // Called when the client containers are filled into the active page
    // print output from server to terminalOutput div
    socket.on('command_output', function(msg) {
        var outputElement = document.getElementById('terminalOutput');
        outputElement.innerHTML += msg.data + '<br>';
        scrollToBottom();
        // Clear terminal if it gets too long
        if (outputElement.innerHTML.length > 10000) {
            outputElement.innerHTML = '';
        }
    })
    // TODO: Use superlink terminal output to monitor number of connected clients
    // possibly display visual representations for each client in a custom container
    // report status of client
    socket.on('command_status', function(msg) { 
        var clientStatusElement = document.getElementById('clientStatus');
        console.log(msg.status)
        if(msg.status === 'active') {
            clientStatusElement.textContent = 'Client running';
            clientStatusElement.classList.replace('inactive', 'active');
            document.getElementById('startClient').textContent = 'Stop Client';
        }if(msg.status === 'inactive'){
            clientStatusElement.textContent = 'Client inactive';
            clientStatusElement.classList.replace('active', 'inactive');
            document.getElementById('startClient').textContent = 'Start Client';
        }if(msg.status === 'completed'){
            if(msg.step === "01"){
                document.getElementById('DetailsExtracted').checked=true;
            } if(msg.step === "02"){
                document.getElementById('DetailsParsed').checked=true;
            } if(msg.step === "03"){
                document.getElementById('NiftiConversion').checked=true;
            } if (msg.step === "04"){
                document.getElementById('RasComplete').checked=true;
            } if (msg.step === "05"){
                document.getElementById('Aligned').checked=true;
            } if (msg.step === "06"){
                document.getElementById('InputsGen').checked=true;
            }
        }
    })  
    
    var activeNodeIDs = new Set();
    socket.on('node_active', function(msg) {
        console.log('Node active:', msg.node_id);
        var nodeImages = document.getElementById('clientMonitor');
        var nodeId = msg.node_id;

        if (!activeNodeIDs.has(nodeId)) {
            activeNodeIDs.add(nodeId);
            var nodeDiv = document.createElement('div');
            nodeDiv.className = 'node';
            nodeDiv.innerHTML = '<img src="/static/comp.png" alt="Node Image"><p>' + nodeId + '</p>';
            nodeImages.appendChild(nodeDiv);
        }
    })
    socket.on('node_inactive', function(msg) {
        console.log('Node inactive:', msg.node_id);
        var nodeImages = document.getElementById('clientMonitor');
        var nodeId = msg.node_id;

        if (activeNodeIDs.has(nodeId)) {
            activeNodeIDs.delete(nodeId);
            var nodeDiv = document.querySelector('.node p');
            if (nodeDiv.textContent === nodeId) {
                nodeDiv.parentElement.remove();
            }
        }
    })
})

