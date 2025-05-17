// src/utils/runTask.js
export async function runTask(startUrl, { method = 'GET', body, pollMs = 1000 } = {}) {
    // kick-off
    const startRes = await fetch(startUrl, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body:   body ? JSON.stringify(body) : undefined,
    });
    if (!startRes.ok) throw new Error(`${startRes.status} ${startRes.statusText}`);
  
    const { id } = await startRes.json();
    if (!id) throw new Error('Task id missing in server response');
  
    // poll until done
    while (true) {
      const res = await fetch(`${CONSTANTS.API_BASE_URL}/data/${id}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  
      const payload = await res.json();
      if (payload.status === 'success' || payload.status === 'finished') {
        return payload.result;        // 👈 final data
      }
      if (payload.status === 'failed') {
        throw new Error(payload.error || 'Task failed on server');
      }
      await new Promise(r => setTimeout(r, pollMs)); // wait & poll again
    }
  }