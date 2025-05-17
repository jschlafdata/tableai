// utils/taskRunner.js ------------------------------------------------
import CONSTANTS from '../../../constants';

export async function runTask(
  endpoint,          // full URL of the “kick-off” request
  fetchOpts = {},    // method / headers / body, if needed
  { pollInterval = 1000, timeout = 30000 } = {}
) {
  const startRes = await fetch(endpoint, fetchOpts);
  if (!startRes.ok) throw new Error(`${startRes.status} ${startRes.statusText}`);

  const data = await startRes.json();

  // ✅ If no id or status, assume eager result and return it directly
  if (!data?.id || !data?.status || data?.result) {
    console.log('[runTask] eager result received:', data);
    return data.result ?? data;
  }

  // ⏳ Otherwise: handle as a background task with polling
  const taskId = data.id;
  const deadline = Date.now() + timeout;

  while (true) {
    const pollRes = await fetch(`${CONSTANTS.API_BASE_URL}/tasks/${taskId}`);
    if (!pollRes.ok) throw new Error(`${pollRes.status} ${pollRes.statusText}`);

    const body = await pollRes.json();

    if ('result' in body) {
      console.log('[runTask] finished task', taskId, body.result);
      return body.result;
    }

    if (Date.now() > deadline) {
      throw new Error(`Task ${taskId} timed out`);
    }

    await new Promise(r => setTimeout(r, pollInterval));
  }
}