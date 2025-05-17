import { useEffect, useState } from 'react'
import CONSTANTS from '../../../constants'

export function useStage0Summary() {
  const [summaryMap, setSummaryMap] = useState({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await fetch(`${CONSTANTS.API_BASE_URL}/query/stage0_summary`)
        const json = await res.json()
        const map = {}
        for (const r of json) {
            if (r.uuid) {
              map[r.uuid] = r
            }
          }
        setSummaryMap(map)
        console.log('summary_map', map)
      } catch (err) {
        console.error('Failed to fetch stage0 summary:', err)
        setError(err)
      } finally {
        setLoading(false)
      }
    }

    load()
  }, [])

  return { summaryMap, loading, error }
}