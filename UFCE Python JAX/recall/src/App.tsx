import { useState } from 'react';
import { useEngine } from './hooks/useEngine';
import './App.css'; 

function App() {
  const { status, logs, baseUrl } = useEngine();
  const [pingResult, setPingResult] = useState<string | null>(null);

  async function handleManualPing() {
    if (!baseUrl) return;
    try {
      const response = await fetch(`${baseUrl}/`);
      const data = await response.json();
      setPingResult(JSON.stringify(data, null, 2));
    } catch (error: unknown) {
      // TypeScript Safe Error Handling
      const errorMessage = error instanceof Error ? error.message : String(error);
      setPingResult("Error: " + errorMessage);
    }
  }

  // Helper for status color
  const getStatusColor = () => {
    if (status === 'ready') return '#d4edda'; // Green
    if (status === 'error') return '#f8d7da'; // Red
    return '#fff3cd'; // Yellow (booting/idle)
  };

  return (
    <div className="container" style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Velocity Recall AI</h1>
      
      {/* 1. STATUS BOX */}
      <div style={{ 
        padding: '15px', 
        borderRadius: '8px', 
        background: getStatusColor(),
        color: '#333',
        border: '1px solid #ccc',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <strong>ENGINE STATUS: </strong> 
        <span style={{ textTransform: 'uppercase', fontWeight: 'bold' }}>{status}</span>
      </div>

      {/* 2. MANUAL PING BUTTON (Only appears when Ready) */}
      {status === 'ready' && (
        <div style={{ textAlign: 'center', marginBottom: '20px' }}>
          <button 
            onClick={handleManualPing} 
            style={{ 
              padding: '10px 20px', 
              fontSize: '16px', 
              cursor: 'pointer',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '5px'
            }}
          >
            📡 Ping Python Engine
          </button>
        </div>
      )}

      {/* 3. SERVER RESPONSE AREA */}
      {pingResult && (
        <div style={{ marginBottom: '20px' }}>
          <h3>Server Response:</h3>
          <pre style={{ background: '#f4f4f4', color: '#333', padding: '15px', borderRadius: '5px' }}>
            {pingResult}
          </pre>
        </div>
      )}

      {/* 4. THE SYSTEM LOGS (The "Black Box") */}
      <div style={{ marginTop: '30px', textAlign: 'left' }}>
        <h4>Debug Logs:</h4>
        <div style={{ 
          background: '#1e1e1e', 
          color: '#00ff00', 
          padding: '15px', 
          borderRadius: '5px', 
          fontFamily: 'monospace',
          height: '200px',
          overflowY: 'auto',
          fontSize: '0.9em'
        }}>
          {logs.length === 0 && <span style={{color: '#666'}}>No logs yet...</span>}
          {logs.map((log, i) => (
            <div key={i} style={{ marginBottom: '5px', borderBottom: '1px solid #333' }}>
              {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;