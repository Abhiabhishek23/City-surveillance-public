import React, {useEffect, useState} from 'react';

export default function AdminStats(){
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(()=>{
    fetch(process.env.REACT_APP_BACKEND_URL ? `${process.env.REACT_APP_BACKEND_URL}/stats/aggregate` : '/stats/aggregate')
      .then(r=>r.json()).then(j=>{setData(j); setLoading(false)}).catch(e=>{setData({error:e.toString()}); setLoading(false)})
  },[])
  if(loading) return <div>Loading...</div>
  return (
    <div style={{padding:20}}>
      <h2>Admin Aggregated Stats</h2>
      <pre style={{background:'#111', color:'#eee', padding:12}}>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}
