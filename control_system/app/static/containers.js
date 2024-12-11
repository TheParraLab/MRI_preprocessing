////////////////////////// TABS //////////////////////////
// This file manages the tabs of the webpage
// Each tab has the following attributes:
// - data-file: the file to be loaded into the container
// - data-requires-password: whether the tab requires a password to be accessed

// Add event listeners to the tabs
function loadContent(file) {
    // Loads the content of the file into the container section
    fetch(file)
        .then(response => response.text())
        .then(html => {
            document.getElementById('containerSection').innerHTML = html; // Insert the requested html file into the 'containerSection' div
        
            if (file === 'client.html'){
                // Dispatch a custom event to notify the containers have been loaded
                // Will start a data scan to provide information to the data container
                document.dispatchEvent(new CustomEvent('containersContentLoaded'));
            }
            document.dispatchEvent(new CustomEvent('containerLoaded'));
        })

        .catch(error => {
            console.warn('Error loading the containers:', error);
        });
}
document.querySelectorAll('.tabs a').forEach(tab => {
    tab.addEventListener('click', function(event) {
        event.preventDefault(); 
        const file = this.getAttribute('data-file');
        const requiresPassword = this.getAttribute('data-requires-password') === 'true';

        // Check if the tab requires a password to be accessed
        if (requiresPassword) {
            const password = prompt('Enter password to access this tab:');
            if (password !== 'admin') { // Password is 'admin', hardcoded for now !!UNSECURE!!
                alert('Incorrect password!');
                return;
            }
        }

        // Load the content of the file into the container section
        loadContent(file);
    });
});