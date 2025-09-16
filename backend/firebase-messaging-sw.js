importScripts('https://www.gstatic.com/firebasejs/9.6.1/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/9.6.1/firebase-messaging-compat.js');

// This is the Firebase configuration for your web app.
// IMPORTANT: Replace this with your actual project config from Firebase Console.
// You can find this by going to Project Settings -> Your apps -> Web app -> Config
const firebaseConfig = {
  apiKey: "AIzaSyCdJOVkxrThmkOxgSvTUsa3z5Hu7lEDCx8",
  authDomain: "ujjain-50cb1.firebaseapp.com",
  projectId: "ujjain-50cb1",
  storageBucket: "ujjain-50cb1.firebasestorage.app",
  messagingSenderId: "317612615610",
  appId: "1:317612615610:web:356c42a095855976b31f83",
  measurementId: "G-XXLLZ9F0QM"
};

const app = firebase.initializeApp(firebaseConfig);
const messaging = firebase.messaging();

// Show a richer notification using data fields and support click-through
messaging.onBackgroundMessage(payload => {
  console.log('Received background message ', payload);
  const n = payload.notification || {};
  const d = payload.data || {};
  const title = n.title || `Alert`;
  const bodyParts = [n.body];
  if (d.event) bodyParts.push(`Type: ${d.event.replace('_',' ')}`);
  if (d.camera_id) bodyParts.push(`Camera: ${d.camera_id}`);
  if (d.count) bodyParts.push(`Count: ${d.count}`);
  const body = bodyParts.filter(Boolean).join(' | ');
  const notificationOptions = {
    body,
    data: { snapshot_url: d.snapshot_url || '' },
    // icon: '/icon.png',
    // badge: '/badge.png'
  };
  return self.registration.showNotification(title, notificationOptions);
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  const url = event.notification?.data?.snapshot_url;
  if (url) {
    event.waitUntil(clients.openWindow(url));
  }
});
