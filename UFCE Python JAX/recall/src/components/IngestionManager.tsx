import React, { useState, useEffect } from 'react';
import { open } from '@tauri-apps/plugin-dialog';

interface IngestionManagerProps {
  baseUrl: string | null;
}

const IngestionManager: React.FC<IngestionManagerProps> = ({ baseUrl }) => {
  const [dbPath, setDbPath] = useState<string>("Scanning...");
  const [dbDim, setDbDim] = useState<string>("-");
  const [queue, setQueue] = useState<string[]>([]);
  const [isIngesting, setIsIngesting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("");

  // 1. Fetch DB Info & Queue on mount (and when baseUrl changes)
  useEffect(() => {
    if (!baseUrl) return;

    const fetchData = async () => {
      try {
        // Get Root Info (DB Path & Dim)
        const rootRes = await fetch(`${baseUrl}/`);
        const rootData = await rootRes.json();
        setDbPath(rootData.db_path || "Unknown");
        setDbDim(rootData.dim || "?");

        // Get Queue Status
        const queueRes = await fetch(`${baseUrl}/get_queue`);
        const queueData = await queueRes.json();
        setQueue(queueData.queue || []);
        setIsIngesting(queueData.is_ingesting);
        setProgress(queueData.progress);
        setStatusText(queueData.status_text);
      } catch (err) {
        console.error("Failed to fetch engine data", err);
      }
    };

    fetchData();
    // Poll updates every 2 seconds if looking at this screen
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [baseUrl]);

  // 2. Handle Adding a Folder
  const handleAddFolder = async () => {
    if (!baseUrl) return;
    try {
      const selected = await open({
        directory: true,
        multiple: false,
      });

      if (selected) {
        await fetch(`${baseUrl}/add_source`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: selected })
        });
        // Trigger immediate refresh
        const res = await fetch(`${baseUrl}/get_queue`);
        const data = await res.json();
        setQueue(data.queue);
      }
    } catch (err) {
      console.error("Error adding folder:", err);
    }
  };

  // 3. Handle Start Ingestion
  const handleStartIngestion = async () => {
    if (!baseUrl) return;
    try {
      await fetch(`${baseUrl}/start_ingestion`, { method: 'POST' });
      setIsIngesting(true);
    } catch (err) {
      console.error("Error starting ingestion:", err);
    }
  };

  // 4. Handle Clear Knowledge
  const handleClearKnowledge = async () => {
    if (!baseUrl) return;
    if (!confirm("Are you sure? This will delete the entire vector database.")) return;
    
    try {
      await fetch(`${baseUrl}/clear_knowledge`, { method: 'POST' });
      window.location.reload(); // Reload app to reset state
    } catch (err) {
      console.error("Error clearing knowledge:", err);
    }
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #333', borderRadius: '8px', background: '#111' }}>
      <h3 style={{ marginTop: 0, color: '#fff' }}>Knowledge Base Manager</h3>
      
      {/* DB STATUS BOX */}
      <div style={{ 
        backgroundColor: '#1a1a1a', 
        padding: '12px', 
        marginBottom: '20px', 
        borderRadius: '6px',
        border: '1px solid #333',
        fontSize: '0.9rem',
        color: '#ccc'
      }}>
        <div style={{ marginBottom: '4px' }}>
          <span style={{ color: '#888' }}>Database Path:</span> 
          <span style={{ marginLeft: '10px', color: '#fff', fontFamily: 'monospace' }}>{dbPath}</span>
        </div>
        <div>
           <span style={{ color: '#888' }}>Vector Dimension:</span>
           <span style={{ marginLeft: '10px', color: '#fff', fontWeight: 'bold' }}>{dbDim}</span> 
           {Number(dbDim) === 384 && <span style={{color:'orange', marginLeft:'8px', fontSize: '0.8em'}}>(Compressed Mode)</span>}
           {Number(dbDim) === 768 && <span style={{color:'#00cc00', marginLeft:'8px', fontSize: '0.8em'}}>(High Quality)</span>}
        </div>
      </div>

      {/* CONTROLS */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button onClick={handleAddFolder} disabled={isIngesting} style={{ padding: '8px 16px', cursor: 'pointer' }}>
          + Add Source Folder
        </button>
        
        <button 
          onClick={handleStartIngestion} 
          disabled={isIngesting || queue.length === 0}
          style={{ 
            padding: '8px 16px', 
            cursor: 'pointer',
            backgroundColor: isIngesting ? '#444' : '#0066cc',
            color: 'white',
            border: 'none',
            borderRadius: '4px'
          }}
        >
          {isIngesting ? 'Ingesting...' : '▶ Start Ingestion'}
        </button>

        <button 
          onClick={handleClearKnowledge}
          disabled={isIngesting}
          style={{ 
            padding: '8px 16px', 
            cursor: 'pointer', 
            backgroundColor: '#330000', 
            color: '#ff4444', 
            border: '1px solid #660000',
            marginLeft: 'auto'
          }}
        >
          Reset Database
        </button>
      </div>

      {/* PROGRESS BAR */}
      {isIngesting && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.85rem', color: '#aaa' }}>
            <span>{statusText}</span>
            <span>{progress.toFixed(1)}%</span>
          </div>
          <div style={{ width: '100%', height: '8px', background: '#333', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: `${progress}%`, height: '100%', background: '#00cc00', transition: 'width 0.3s ease' }} />
          </div>
        </div>
      )}

      {/* QUEUE LIST */}
      <div style={{ marginTop: '10px' }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>Ingestion Queue</h4>
        {queue.length === 0 ? (
          <div style={{ color: '#555', fontStyle: 'italic', fontSize: '0.9rem' }}>Queue is empty. Add a folder to begin.</div>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {queue.map((path, i) => (
              <li key={i} style={{ 
                padding: '8px', 
                marginBottom: '4px', 
                background: '#222', 
                borderRadius: '4px', 
                fontFamily: 'monospace', 
                fontSize: '0.85rem',
                color: '#ddd'
              }}>
                {path}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default IngestionManager;