import CONSTANTS from '../../../constants'
import { runTask } from './taskRunner'

export async function fetchDropboxFolder(currentPath, sortDir) {
  const segments = String(currentPath)
    .split('/')
    .filter(Boolean)
    .map(encodeURIComponent)
    .join('/')
  const url = segments
    ? `${CONSTANTS.API_BASE_URL}/${CONSTANTS.DROPBOX_FOLDERS}/${segments}`
    : `${CONSTANTS.API_BASE_URL}/${CONSTANTS.DROPBOX_FOLDERS}`

  const data = await runTask(url)

  const itemsArr = Array.isArray(data) ? data
    : Array.isArray(data.folder_items) ? data.folder_items
    : (() => { throw new Error('Bad format') })()

  return itemsArr.sort((a, b) =>
    sortDir === 'asc' ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name)
  )
}

export async function fetchStage0Summary() {
  const res = await fetch(`${CONSTANTS.API_BASE_URL}/query/stage0_summary`)
  return res.json()
}

export async function fetchSyncHistory() {
  const res = await fetch(`${CONSTANTS.API_BASE_URL}/${CONSTANTS.DROPBOX_BASE}/sync/history`)
  const records = await res.json()
  const map = {}
  records.forEach(r => { map[r.path_lower] = r })
  return map
}