export default function formatBytes(bytes = 0) {
    if (bytes < 1024) return `${bytes} B`
    const k = 1024, dm = 2, sizes = ['KB','MB','GB','TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(dm)} ${sizes[i]}`
  }