import { TABLEAI_SERVICES } from './serviceBaseUrls';

const BASE = TABLEAI_SERVICES.base;

export const TABLEAI_SERVICE_ENDPOINTS = {
  query: {
    trigger: `${BASE}${TABLEAI_SERVICES.query}/trigger`,
    classifyLabels: `${BASE}${TABLEAI_SERVICES.query}/classify/labels`,
    labelResult: `${BASE}${TABLEAI_SERVICES.query}/label-result`, // example
    records: `${BASE}${TABLEAI_SERVICES.query}/records`,
    filterMetadata: `${BASE}${TABLEAI_SERVICES.query}/filters/metadata`
  },

  dropbox: {
    listFiles: `${BASE}${TABLEAI_SERVICES.dropbox}/files`,
    syncFile: `${BASE}${TABLEAI_SERVICES.dropbox}/sync`,
    fileInfo: (fileId) => `${BASE}${TABLEAI_SERVICES.dropbox}/files/${fileId}`,
  },

  tableai: {
    extract: `${BASE}${TABLEAI_SERVICES.tableai}/extract`,
    analyze: `${BASE}${TABLEAI_SERVICES.tableai}/analyze`,
    process: (stage, fileId) => `${BASE}${TABLEAI_SERVICES.tableai}/process/${stage}/${fileId}`,
  },

  ui: {
    postClassifyLabels: `${BASE}${TABLEAI_SERVICES.ui}/post/classify/labels`,
    feedback: `${BASE}${TABLEAI_SERVICES.ui}/feedback`,
  }
};