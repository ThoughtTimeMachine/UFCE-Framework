import { useState } from 'react';

interface SearchInterfaceProps {
  baseUrl: string | null;
  status: string;
}

export function SearchInterface({ baseUrl, status }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault(); // Stop page reload
    if (!baseUrl || !query.trim()) return;

    setLoading(true);
    setResults(null); // Clear previous results

    try {
      // 1. Send POST request to Python /search endpoint
      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query }),
      });

      // 2. Parse the result
      const data = await response.json();
      
      // The backend returns a pre-formatted string in "results"
      setResults(data.results); 
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setResults(`Error: ${msg}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto' }}>
      {/* SEARCH INPUT AREA */}
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask your archive..."
          disabled={status !== 'ready'}
          style={{
            flex: 1,
            padding: '15px',
            fontSize: '1.1rem',
            borderRadius: '8px',
            border: '1px solid #444',
            background: '#222',
            color: 'white',
            outline: 'none'
          }}
        />
        <button 
          type="submit" 
          disabled={status !== 'ready' || loading}
          style={{
            padding: '0 30px',
            fontSize: '1rem',
            fontWeight: 'bold',
            background: status === 'ready' ? '#007bff' : '#444',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: status === 'ready' ? 'pointer' : 'not-allowed',
            opacity: loading ? 0.7 : 1
          }}
        >
          {loading ? 'Thinking...' : 'Search'}
        </button>
      </form>

      {/* RESULTS DISPLAY AREA */}
      {results && (
        <div className="results-container" style={{ 
          textAlign: 'left', 
          background: '#1a1a1a', 
          padding: '25px', 
          borderRadius: '12px',
          border: '1px solid #333',
          boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
        }}>
          <h3 style={{ marginTop: 0, color: '#007bff', borderBottom: '1px solid #333', paddingBottom: '10px', marginBottom: '15px' }}>
            Retrieval Results
          </h3>
          <div style={{ 
            fontSize: '1rem', 
            lineHeight: '1.6', 
            whiteSpace: 'pre-wrap', // CRITICAL: Preserves Python's formatting/newlines
            color: '#e0e0e0',
            fontFamily: 'monospace' // Optional: Makes data alignment look nicer
          }}>
            {results}
          </div>
        </div>
      )}
    </div>
  );
}