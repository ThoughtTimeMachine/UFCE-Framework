import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export function useEngine() {
  const [status, setStatus] = useState('booting');
  const [logs, setLogs] = useState<string[]>([]);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null;
    let unlistenFn: (() => void) | undefined;

    // TASK 1: SETUP LOGS (Don't await this blocking the rest!)
    listen<string>('plugin:process|stdout', (event) => {
        const line = event.payload;
        setLogs((prev) => [...prev.slice(-49), line]);
    }).then((fn) => {
        unlistenFn = fn;
    }).catch(err => console.warn("Log listener failed (non-critical):", err));

    // TASK 2: POLL FOR PORT (Start Immediately!)
    console.log("React: Starting to poll for engine port...");
    
    intervalId = setInterval(async () => {
      try {
        // Ask Rust for the port number
        const port = await invoke<number>('get_api_port');
        
        // Log to browser console so we can see it working
        console.log("React: Rust reported port:", port);

        if (port && port > 0) {
          const url = `http://127.0.0.1:${port}`;
          console.log("React: Connecting to", url);
          
          setBaseUrl(url);
          setStatus('ready');
          
          // Stop polling once we have the port
          if (intervalId) clearInterval(intervalId);
        }
      } catch (err) {
        console.error("React: Error asking Rust for port:", err);
      }
    }, 1000); // Check every 1 second

    // Cleanup
    return () => {
      if (intervalId) clearInterval(intervalId);
      if (unlistenFn) unlistenFn();
    };
  }, []);

  return { status, logs, baseUrl };
}