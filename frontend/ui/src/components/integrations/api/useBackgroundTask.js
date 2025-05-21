// src/hooks/useBackgroundTask.js
import { useState, useEffect, useCallback } from 'react'

export function useBackgroundTask({
  pollUrlBase = '/tasks',
  pollInterval = 1500,
  onComplete,
  onError
} = {}) {
  const [taskId, setTaskId] = useState(null)
  const [isRunning, setIsRunning] = useState(false)

  const startTask = useCallback(async (startUrl, options = {}) => {
    setIsRunning(true)
    try {
      const res = await fetch(startUrl, {
        method: options.method || 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {})
        },
        body: options.body ? JSON.stringify(options.body) : undefined
      })
      const data = await res.json()
      if (!data.id) throw new Error('No task ID returned')
      setTaskId(data.id)
    } catch (err) {
      console.error('Failed to start task:', err)
      setIsRunning(false)
      onError?.(err)
    }
  }, [onError])

  // Polling effect
  useEffect(() => {
    if (!taskId) return

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${pollUrlBase}/${taskId}`)
        const data = await res.json()

        if (data.status === 'completed' || data.status === 'finished') {
          clearInterval(interval)
          setTaskId(null)
          setIsRunning(false)
          onComplete?.(data.result ?? data)
        } else if (data.status === 'failed') {
          clearInterval(interval)
          setTaskId(null)
          setIsRunning(false)
          onError?.(new Error(data.error || 'Task failed'))
        }
      } catch (err) {
        clearInterval(interval)
        setTaskId(null)
        setIsRunning(false)
        console.error('Polling failed:', err)
        onError?.(err)
      }
    }, pollInterval)

    return () => clearInterval(interval)
  }, [taskId, pollInterval, pollUrlBase, onComplete, onError])

  return { startTask, isRunning, taskId }
}