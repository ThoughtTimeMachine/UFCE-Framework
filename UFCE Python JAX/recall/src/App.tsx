import { useState } from 'react';
import { useEngine } from './hooks/useEngine';
import { SearchInterface } from './components/SearchInterface';
import IngestionManager from './components/IngestionManager';
import './App.css'; 

function App() {
  const { status, logs, baseUrl } = useEngine();
  const [showLogs, setShowLogs] = useState(false);

  // Helper for status color
  const getStatusColor = () => {
    if (status === 'ready') return 'rgba(0, 255, 0, 0.1)'; // Greenish background
    if (status === 'error') return 'rgba(255, 0, 0, 0.1)'; // Reddish background
    return 'rgba(255, 255, 0, 0.1)'; // Yellowish background
  };

  const getStatusTextColor = () => {
    if (status === 'ready') return '#00cc00';
    if (status === 'error') return '#ff4444';
    return '#cccc00';
  };

  return (
    <div className="container" style={{ padding: '40px 20px', fontFamily: 'Inter, system-ui, sans-serif', maxWidth: '1000px', margin: '0 auto' }}>
      
      {/* HEADER */}
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ marginBottom: '15px', fontSize: '2.5rem', letterSpacing: '-1px' }}>Velocity Recall</h1>
        <div style={{ 
          display: 'inline-block',
          padding: '6px 16px', 
          borderRadius: '20px', 
          fontSize: '0.85rem',
          fontWeight: 600,
          background: getStatusColor(),
          color: getStatusTextColor(),
          border: `1px solid ${getStatusTextColor()}`,
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          ● Engine Status: {status}
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <main>
        {/* 1. Search Interface (Always visible) */}
        <SearchInterface baseUrl={baseUrl} status={status} />
        
        {/* 2. Ingestion Manager (Only visible when engine is ready) */}
        {status === 'ready' && (
          <IngestionManager baseUrl={baseUrl} />
        )}
      </main>

      {/* FOOTER / DEBUGGER */}
      <footer style={{ marginTop: '80px', borderTop: '1px solid #333', paddingTop: '20px', textAlign: 'center' }}>
        <button 
          onClick={() => setShowLogs(!showLogs)}
          style={{ 
            background: 'none', 
            border: 'none', 
            color: '#666', 
            cursor: 'pointer', 
            fontSize: '0.85rem', 
            textDecoration: 'underline' 
          }}
        >
          {showLogs ? 'Hide System Logs' : 'Show Debug Logs'}
        </button>
        
        {showLogs && (
          <div style={{ 
            marginTop: '20px',
            background: '#0d0d0d', 
            color: '#00ff00', 
            padding: '20px', 
            borderRadius: '8px', 
            fontFamily: 'monospace',
            height: '200px',
            overflowY: 'auto',
            fontSize: '0.85rem',
            textAlign: 'left',
            border: '1px solid #333'
          }}>
            {logs.length === 0 && <span style={{color: '#666'}}>Waiting for engine...</span>}
            {logs.map((log, i) => (
              <div key={i} style={{ marginBottom: '4px', borderBottom: '1px solid #222' }}>
                {log}
              </div>
            ))}
          </div>
        )}
      </footer>
    </div>
  );
}

export default App;