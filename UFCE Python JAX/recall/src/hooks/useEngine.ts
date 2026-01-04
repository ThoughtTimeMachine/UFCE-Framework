import { useState, useEffect, useRef } from 'react';
import { Command, Child } from '@tauri-apps/plugin-shell';

// Define the shape of the Close Event from Tauri
interface CloseEvent {
  code: number;
  signal: number | null;
}

export function useEngine() {
  const [status, setStatus] = useState<string>('idle'); 
  const [logs, setLogs] = useState<string[]>([]);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const childProcess = useRef<Child | null>(null);

  useEffect(() => {
    let isMounted = true;

    // Helper to verify if the sidecar is responding
    async function checkHealth(url: string) {
      let attempts = 0;
      while (attempts < 10 && isMounted) {
        try {
          const res = await fetch(`${url}/`);
          if (res.ok) {
            setStatus('ready');
            addLog('🟢 Connection established! Engine is online.');
            return;
          }
        } catch {
           // Connection refused? Just wait and retry.
        }
        await new Promise(r => setTimeout(r, 500));
        attempts++;
      }
      if (isMounted) setStatus('timeout');
    }

    // Helper to append logs safely
    function addLog(msg: string) {
      setLogs(prev => [...prev.slice(-9), msg]); 
    }

    async function startEngine() {
      try {
        setStatus('booting');
        addLog('🚀 Spawning Engine (Waiting for Port)...');

        // This path must match 'externalBin' in tauri.conf.json
        const command = Command.sidecar('binaries/velocity-engine');

        // EVENT 1: STDOUT (The Python engine talking)
        command.stdout.on('data', (line: string) => {
          if (isMounted) {
            console.log('[PY]', line);
            // Check for the "Magic String" we printed in main.py
            if (line.includes('VELOCITY_PORT:')) {
              const detectedPort = line.split('VELOCITY_PORT:')[1].trim();
              const portNum = parseInt(detectedPort);
              
              if (!isNaN(portNum)) {
                const url = `http://localhost:${portNum}`;
                setBaseUrl(url);
                addLog(`🎯 Port Detected: ${portNum}`);
                checkHealth(url); 
              }
            }
          }
        });

        // EVENT 2: CLOSE (The engine stopped)
        command.on('close', (data: unknown) => {
          if (isMounted) {
            // Cast 'data' to the CloseEvent interface we defined above
            const event = data as CloseEvent; 
            addLog(`⚠️ Engine stopped with code ${event.code}`);
            setStatus('error');
          }
        });

        // EVENT 3: ERROR (Startup failure)
        command.on('error', (error: unknown) => {
          if (isMounted) {
            addLog(`❌ Engine error: ${String(error)}`);
            setStatus('error');
          }
        });

        const child = await command.spawn();
        childProcess.current = child;

      } catch (err: unknown) {
        // Handle generic JS errors during spawn
        const message = err instanceof Error ? err.message : String(err);
        addLog(`❌ Failed to spawn: ${message}`);
        setStatus('error');
      }
    }

    startEngine();

    // CLEANUP: Kill the process when the React component unmounts
    return () => {
      isMounted = false;
      if (childProcess.current) {
        childProcess.current.kill();
      }
    };
  }, []);

  return { status, logs, baseUrl };
}