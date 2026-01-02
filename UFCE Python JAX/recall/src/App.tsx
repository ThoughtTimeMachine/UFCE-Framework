import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

function App() {
  const [port, setPort] = useState<number | null>(null);
  const [response, setResponse] = useState<string>("");

  useEffect(() => {
    const connect = async () => {
      // Poll Rust for the port (retry logic)
      for (let i = 0; i < 20; i++) {
        const p = await invoke<number>("get_api_port");
        if (p !== 0) { 
            console.log("Connected on port:", p);
            setPort(p); 
            return; 
        }
        await new Promise(r => setTimeout(r, 500));
      }
    };
    connect();
  }, []);

  const testQuery = async () => {
    if (!port) return;
    try {
        const res = await fetch(`http://127.0.0.1:${port}/query`, {
            method: "POST",
            body: JSON.stringify({ query: "Hello from React!" }),
            headers: { "Content-Type": "application/json" }
        });
        const data = await res.json();
        setResponse(JSON.stringify(data, null, 2));
    } catch (e) {
        console.error(e);
        setResponse("Error connecting to Python");
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Velocity Recall</h1>
      <p>Status: {port ? <span style={{color: 'green'}}>● Online (Port {port})</span> : <span style={{color: 'red'}}>● Connecting...</span>}</p>
      
      <button onClick={testQuery} disabled={!port} style={{ padding: "10px 20px", fontSize: "16px" }}>
        Test Neural Link
      </button>

      <pre style={{ marginTop: 20, background: "#f0f0f0", padding: 10, borderRadius: 5 }}>
        {response || "Waiting for input..."}
      </pre>
    </div>
  );
}
export default App;