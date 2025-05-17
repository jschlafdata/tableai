export function normalizeRecords(input = {}) {
    const {
      dropbox_sync_records = [],
      extraction_metadata = {},
    } = input;
  
    return dropbox_sync_records.map((record) => {
      const enriched = extraction_metadata?.[record.dropbox_safe_id] || {};
      const parsedMeta = safeJson(record.extraction_metadata_json || '{}');
      const completedStages = safeJson(record.completed_stages_json || '[]');
      const storedStages = safeJson(record.stored_stages_json || '[]');
  
      return {
        ...record,
        ...enriched,
        ...parsedMeta,
        completed_stages: completedStages,
        stored_stages: storedStages,
      };
    });
  }
  
  export function safeJson(input, fallback = {}) {
    try {
      return JSON.parse(input);
    } catch (err) {
      console.warn('Failed to parse JSON:', input);
      return fallback;
    }
  }