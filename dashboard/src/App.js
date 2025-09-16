import React, { useState, useEffect, useRef, memo } from 'react';
// Firebase Web compat (simple to drop-in)
import firebase from 'firebase/compat/app';
import 'firebase/compat/auth';
import './App.css';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [status, setStatus] = useState('Connecting...');
  const [toast, setToast] = useState(null);
  const [idToken, setIdToken] = useState(() => localStorage.getItem('admin_id_token') || '');
  const [uid, setUid] = useState(() => localStorage.getItem('admin_uid') || '');
  const [past, setPast] = useState([]); // Past 24h snippets (admin)
  const lastHighIdRef = useRef(null);
  const THRESHOLD = 9;

  const AlertCard = memo(function AlertCard({ a }){
    const imgRef = useRef(null);
    useEffect(()=>{
      const el = imgRef.current; if(!el) return;
      const onLoad = ()=> el.classList.add('loaded');
      const onError = ()=> {
        // one-time cache-bust retry
        if (!el.dataset.retried && a.image_url) {
          el.dataset.retried = '1';
          const url = new URL(a.image_url, window.location.origin);
          url.searchParams.set('_', Date.now());
          el.src = url.toString();
        } else {
          el.classList.add('loaded');
        }
      };
      el.addEventListener('load', onLoad); el.addEventListener('error', onError);
      if (el.complete) {
        try { el.decode ? el.decode().then(()=> el.classList.add('loaded')).catch(()=> el.classList.add('loaded')) : el.classList.add('loaded'); } catch { el.classList.add('loaded'); }
      }
      return ()=>{ el.removeEventListener('load', onLoad); el.removeEventListener('error', onError); };
    }, [a.image_url]);
    return (
      <div className={`alert-card ${a.type} ${(a.type === 'overcrowding' && Number(a.count || 0) >= THRESHOLD) ? 'high' : ''}`}>
        <h3>{a.type.replace('_', ' ')}</h3>
        <p>Camera: {a.camera_id} | Count: {a.count}</p>
        <p>{new Date(a.timestamp).toLocaleString()}</p>
        {a.image_url ? (
          <>
            <div className="skeleton" style={{width:'100%', height:160, borderRadius:10, marginTop:8}} />
            <img ref={imgRef} src={a.image_url} alt="Alert snapshot" loading="lazy" decoding="async" style={{marginTop:-160, display:'block'}} />
          </>
        ) : null}
      </div>
    );
  });

  // Initialize Firebase app once
  useEffect(() => {
    if (!firebase.apps.length) {
      firebase.initializeApp({
        apiKey: 'AIzaSyCdJOVkxrThmkOxgSvTUsa3z5Hu7lEDCx8',
        authDomain: 'ujjain-50cb1.firebaseapp.com',
        projectId: 'ujjain-50cb1',
        storageBucket: 'ujjain-50cb1.firebasestorage.app',
        messagingSenderId: '317612615610',
        appId: '1:317612615610:web:356c42a095855976b31f83'
      });
    }
  }, []);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        // Try admin endpoint first
        const headers = idToken ? { 'Authorization': `Bearer ${idToken}` } : {};
  let resp = await fetch('/alerts?since_hours=24&limit=200', { headers });
        let arr = [];
        if (resp.status === 403 || resp.status === 401) {
          // Not admin; use client snippets endpoint
          resp = await fetch('/client_alerts?since_hours=24&limit=200', { headers });
          if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
          const j = await resp.json();
          // normalize to dashboard shape
          arr = (Array.isArray(j) ? j : []).map(a => ({
            id: a.id,
            image_url: a.snapshot_url,
            type: a.event,
            timestamp: a.timestamp,
            camera_id: a.camera_id,
            count: a.count,
          }));
          // Client mode: clear past-day list
          setPast([]);
        } else if (resp.ok) {
          const j = await resp.json();
          arr = Array.isArray(j) ? j : [];
          // Admin mode: fetch past 24h snippets
          try {
            const pastResp = await fetch('/alerts?since_hours=24&limit=200', { headers });
            if (pastResp.ok) {
              const pj = await pastResp.json();
              setPast(Array.isArray(pj) ? pj : []);
            } else {
              setPast([]);
            }
          } catch {
            setPast([]);
          }
        } else {
          throw new Error(`HTTP Error: ${resp.status}`);
        }
        setAlerts(arr);
        setStatus('Connected');

        // Show a visible warning when a new high-severity overcrowding alert arrives
        const high = arr.find(a => a && a.type === 'overcrowding' && Number(a.count || 0) >= THRESHOLD);
        if (high && high.id !== lastHighIdRef.current) {
          lastHighIdRef.current = high.id;
          setToast(`⚠️ Overcrowding detected: ${high.count} people at ${high.camera_id}`);
          // Auto-hide after 6s
          setTimeout(() => setToast(null), 6000);
        }
      } catch (error) {
        console.error("Failed to fetch alerts:", error);
        setStatus('Disconnected');
      }
    };

    fetchAlerts(); // Fetch once on load
    const liveInterval = setInterval(fetchAlerts, 5000); // Poll every 5 seconds for live

    return () => {
      clearInterval(liveInterval);
    };
  }, [idToken]);

  const signedIn = !!idToken;
  return (
    <div className="App">
      {toast && (
        <div className="toast warning" role="alert" aria-live="assertive">
          {toast}
        </div>
      )}
      <header className="App-header">
        <h1>City Intelligence • Admin</h1>
        <div style={{marginTop:6}}>
          <a href="/client_ui/mocks/index.html" style={{color:'#b8c0cc', textDecoration:'underline'}}>UI mocks</a>
        </div>
        <p className="subtitle">Live Alerts</p>
        <p className={`status-${status.toLowerCase()}`}>Status: {status}</p>
        {!signedIn && (
          <div style={{marginTop: '8px'}}>
            <input
              placeholder="Paste Firebase ID token for admin"
              value={idToken}
              onChange={(e) => setIdToken(e.target.value)}
              style={{width: '60%'}}
            />
            <button onClick={() => localStorage.setItem('admin_id_token', idToken)} style={{marginLeft: '8px'}}>Save</button>
            <button onClick={() => { setIdToken(''); localStorage.removeItem('admin_id_token'); }} style={{marginLeft: '8px'}}>Clear</button>
          </div>
        )}
      </header>
      <div style={{margin:'10px 0'}}>
        {!signedIn && (
          <button onClick={async () => {
            const provider = new firebase.auth.GoogleAuthProvider();
            const cred = await firebase.auth().signInWithPopup(provider);
            const user = cred.user;
            const tok = await cred.user.getIdToken();
            setIdToken(tok);
            localStorage.setItem('admin_id_token', tok);
            if (user && user.uid) {
              setUid(user.uid);
              localStorage.setItem('admin_uid', user.uid);
            }
          }}>Sign in with Google (Admin)</button>
        )}
        {signedIn && (
          <>
            <button onClick={async ()=>{
              await firebase.auth().signOut();
              setIdToken('');
              setUid('');
              localStorage.removeItem('admin_id_token');
              localStorage.removeItem('admin_uid');
            }}>Sign out</button>
            {uid && (
              <span style={{marginLeft:'12px'}}>
                UID: <code>{uid}</code>
                <button style={{marginLeft:'6px'}} onClick={()=>navigator.clipboard.writeText(uid)}>Copy UID</button>
              </span>
            )}
            {idToken && (
              <button style={{marginLeft:'8px'}} onClick={()=>navigator.clipboard.writeText(idToken)}>Copy ID token</button>
            )}
          </>
        )}
      </div>
      <div className="alerts-container">
        {alerts.length > 0 ? (
          alerts.map(a => (
            <AlertCard key={a.id} a={a} />
          ))
        ) : (
          <p>No alerts to display.</p>
        )}
      </div>

      {past && past.length > 0 && (
        <section className="past-section">
          <h2>Past 24 hours</h2>
          <div className="past-container">
            {past.map(p => {
              const ref = React.createRef();
              return (
              <div key={`p-${p.id}`} className="snippet-card">
                {p.image_url && (
                  <img ref={ref} alt="Snapshot" src={p.image_url} loading="lazy" decoding="async"
                   onLoad={(e)=> e.currentTarget.classList.add('loaded')}
                   onError={(e)=> e.currentTarget.classList.add('loaded')} />
                )}
                <div className="snippet-meta">
                  <span className="loc">{p.camera_id}</span>
                  <span className="time">{p.timestamp ? new Date(p.timestamp).toLocaleString() : ''}</span>
                </div>
              </div>
            )})}
          </div>
        </section>
      )}
    </div>
  );
}

export default App;