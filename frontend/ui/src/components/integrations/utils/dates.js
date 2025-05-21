// Helpers to humanize bytes & dates
export default function formatDate(iso) {
    if (!iso) return '—'
    const d = new Date(iso)
    return (
        d.toLocaleDateString(undefined, { day:'2-digit', month:'short', year:'numeric' }) +
        ' ' +
        d.toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit' })
    )
}