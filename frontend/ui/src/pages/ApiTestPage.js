import React, { useState } from 'react'

import CONSTANTS from '../constants'

const API_BASE = CONSTANTS.API_BASE_URL
const DROPBOX_FOLDERS = CONSTANTS.DROPBOX_FOLDERS

export default function ApiTestPage() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const callApi = async (path, method = 'GET', body = null) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: body ? JSON.stringify(body) : undefined
      })

      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2>📡 API Test Dashboard</h2>

      <button onClick={() => callApi('/dropbox/folders')}>
        GET ALL FOLDERS
      </button>

      <button
        onClick={() =>
          callApi('/dropbox/sync', 'POST', {
            category: 'dropbox',
            resource_id: 'some-id',
            force_refresh: true
          })
        }
      >
        POST /dropbox/sync
      </button>

      <button onClick={() => callApi('/query/records')}>
        GET /query/records
      </button>

      <button onClick={() => callApi('/query/records?merge=true')}>
        GET /query/records?merge=true
      </button>

      <button onClick={() => callApi('/health')}>
        GET /health
      </button>

      <hr />

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {result && (
        <pre
          style={{
            maxHeight: '400px',
            overflow: 'auto',
            background: '#f0f0f0',
            padding: '1rem'
          }}
        >
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}