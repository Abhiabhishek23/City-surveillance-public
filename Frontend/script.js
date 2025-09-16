<script type="module"> // Import the functions you need from the SDKs you need import { initializeApp } from "https://www.gstatic.com/firebasejs/12.2.1/firebase-app.js"; import { getAnalytics } from "https://www.gstatic.com/firebasejs/12.2.1/firebase-analytics.js"; // TODO: Add SDKs for Firebase products that you want to use // https://firebase.google.com/docs/web/setup#available-libraries // Your web app's Firebase configuration // For Firebase JS SDK v7.20.0 and later, measurementId is optional const firebaseConfig = { apiKey: "AIzaSyCdJOVkxrThmkOxgSvTUsa3z5Hu7lEDCx8", authDomain: "ujjain-50cb1.firebaseapp.com", projectId: "ujjain-50cb1", storageBucket: "ujjain-50cb1.firebasestorage.app", messagingSenderId: "317612615610", appId: "1:317612615610:web:356c42a095855976b31f83", measurementId: "G-XXLLZ9F0QM" }; // Initialize Firebase const app = initializeApp(firebaseConfig); const analytics = getAnalytics(app); </script>
// script.js

// --- PASTE YOUR FIREBASE CONFIGURATION HERE ---
const firebaseConfig = {
    apiKey: "AIza...",
    authDomain: "your-project-id.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project-id.appspot.com",
    messagingSenderId: "12345...",
    appId: "1:12345..."
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const messaging = firebase.messaging();

// --- DOM ELEMENTS ---
const statusSpan = document.getElementById('connection-status');
const alertsContainer = document.getElementById('alerts-container');
const notificationBtn = document.getElementById('notification-btn');

// --- APP LOGIC ---
const BACKEND_URL = 'http://127.0.0.1:8000/alerts';

// Function to fetch alerts from the backend
async function fetchAlerts() {
    try {
        const response = await fetch(BACKEND_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const alerts = await response.json();
        statusSpan.textContent = 'Connected';
        statusSpan.style.color = '#4caf50'; // Green for connected
        displayAlerts(alerts);
    } catch (error) {
        console.error('Failed to fetch alerts:', error);
        statusSpan.textContent = 'Disconnected';
        statusSpan.style.color = '#f44336'; // Red for disconnected
    }
}

// Function to display alerts on the page
function displayAlerts(alerts) {
    alertsContainer.innerHTML = ''; // Clear existing alerts
    alerts.forEach(alert => {
        const card = document.createElement('div');
        card.className = `alert-card ${alert.event}`;

        const eventType = alert.event.replace('_', ' ');
        const time = new Date(alert.created_at).toLocaleString();

        card.innerHTML = `
            <div class="alert-header">
                <span class="alert-title">${eventType}</span>
                <span class="alert-time">${time}</span>
            </div>
            <p class="alert-details">Camera: ${alert.camera_id} | Count: ${alert.count}</p>
            ${alert.snapshot ? `<img src="http://127.0.0.1:8000/${alert.snapshot}" class="alert-snapshot" alt="Snapshot">` : ''}
        `;
        alertsContainer.prepend(card); // Add new alerts to the top
    });
}


// --- PUSH NOTIFICATION LOGIC ---
notificationBtn.addEventListener('click', () => {
    console.log('Requesting permission for notifications...');
    Notification.requestPermission().then((permission) => {
        if (permission === 'granted') {
            console.log('Notification permission granted.');
            notificationBtn.textContent = 'Notifications Enabled';
            notificationBtn.disabled = true;
            
            // Get the token
            messaging.getToken({ vapidKey: '--- PASTE YOUR VAPID KEY HERE ---' })
.then((currentToken) => {
    if (currentToken) {
        console.log('FCM Token:', currentToken);
        // --- THIS IS THE NEW CODE ---
        const formData = new FormData();
        formData.append('token', currentToken);

        fetch('http://127.0.0.1:8000/register_fcm_token', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => console.log('Backend response:', data))
        .catch(error => console.error('Error sending token to backend:', error));
        // --- END OF NEW CODE ---

        alert('Notifications enabled! Your token has been sent to the backend.');
    }
    // ... rest of the code
                } else {
                    console.log('No registration token available. Request permission to generate one.');
                }
            }).catch((err) => {
                console.log('An error occurred while retrieving token. ', err);
            });
        } else {
            console.log('Unable to get permission to notify.');
        }
    });
});

// --- INITIALIZE ---
// Fetch alerts when the page loads, and then every 5 seconds
fetchAlerts();
setInterval(fetchAlerts, 5000);