// src/components/pdf/PdfViewer.tsx
import { useState, useRef } from "react";
import { ChevronLeft, ChevronRight, FileText, RefreshCw, Eye, EyeOff } from "lucide-react";
import PageLayout from "../layouts/PageLayout";
import type { User } from "../../types/user";
import type { CSSProperties } from "react";

// Types
interface PdfMetadata {
  id: string;
  name: string;
  size: number;
  source: string;
  path?: string;
  synced_at: string;
  has_vision_results?: boolean;
  table_count?: number;
}

interface PdfViewerProps {
  user: User | null;
}

// Mock data for demonstration
const MOCK_PDFS: PdfMetadata[] = [
  {
    id: "pdf-001",
    name: "Financial Report Q4 2024.pdf",
    size: 2457600,
    source: "dropbox",
    path: "/reports/financial",
    synced_at: "2024-12-15T10:30:00Z",
    has_vision_results: true,
    table_count: 5,
  },
  {
    id: "pdf-002",
    name: "Marketing Strategy 2025.pdf",
    size: 1843200,
    source: "google_drive",
    path: "/marketing/strategy",
    synced_at: "2024-12-20T14:15:00Z",
    has_vision_results: false,
  },
  {
    id: "pdf-003",
    name: "Contract_Amendment_v2.pdf",
    size: 512000,
    source: "s3",
    path: "/legal/contracts",
    synced_at: "2024-12-18T09:00:00Z",
    has_vision_results: true,
    table_count: 2,
  },
];

const PdfViewer: React.FC<PdfViewerProps> = ({ user }) => {
  // State
  const [pdfs] = useState<PdfMetadata[]>(MOCK_PDFS);
  const [selectedPdf, setSelectedPdf] = useState<PdfMetadata | null>(MOCK_PDFS[0]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [numPages] = useState(10); // Mock total pages
  const [showVisionResults, setShowVisionResults] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSource, setSelectedSource] = useState("all");
  const pdfContainerRef = useRef<HTMLDivElement>(null);

  // Filter PDFs
  const filteredPdfs = pdfs.filter((pdf) => {
    const matchesSearch = pdf.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSource = selectedSource === "all" || pdf.source === selectedSource;
    return matchesSearch && matchesSource;
  });

  // Navigation handlers
  const goToPrevious = () => {
    if (selectedFileIndex > 0) {
      const newIndex = selectedFileIndex - 1;
      setSelectedFileIndex(newIndex);
      setSelectedPdf(filteredPdfs[newIndex]);
      setPageNumber(1);
    }
  };

  const goToNext = () => {
    if (selectedFileIndex < filteredPdfs.length - 1) {
      const newIndex = selectedFileIndex + 1;
      setSelectedFileIndex(newIndex);
      setSelectedPdf(filteredPdfs[newIndex]);
      setPageNumber(1);
    }
  };

  const goToPrevPage = () => setPageNumber((p) => Math.max(p - 1, 1));
  const goToNextPage = () => setPageNumber((p) => Math.min(p + 1, numPages));

  const handleRefresh = () => {
    console.log("Refreshing PDFs...");
  };

  const styles: Record<string, CSSProperties> = {
    container: {
      padding: "16px",
    },
    statsHeader: {
      padding: "16px",
      backgroundColor: "#f5f5f5",
      borderRadius: "8px",
      marginBottom: "16px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    statsInfo: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      fontSize: "16px",
      fontWeight: 600,
      color: "#374151",
    },
    filterContainer: {
      padding: "16px",
      backgroundColor: "white",
      borderRadius: "8px",
      marginBottom: "16px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    },
    filterRow: {
      display: "flex",
      gap: "12px",
      flexWrap: "wrap" as const,
    },
    searchInput: {
      flex: 1,
      minWidth: "200px",
      padding: "8px 12px",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      fontSize: "14px",
      outline: "none",
    },
    selectInput: {
      padding: "8px 12px",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      fontSize: "14px",
      outline: "none",
      backgroundColor: "white",
      cursor: "pointer",
    },
    mainContent: {
      display: "flex",
      gap: "16px",
    },
    leftPanel: {
      flex: 1,
      backgroundColor: "white",
      borderRadius: "8px",
      padding: "16px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    },
    rightPanel: {
      flex: 2,
      backgroundColor: "white",
      borderRadius: "8px",
      padding: "16px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    },
    navigationHeader: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "16px",
    },
    navButtons: {
      display: "flex",
      gap: "8px",
    },
    navButton: {
      padding: "6px",
      backgroundColor: "white",
      border: "1px solid #d1d5db",
      borderRadius: "4px",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    fileInfo: {
      marginBottom: "24px",
    },
    fileTitle: {
      fontSize: "18px",
      fontWeight: 600,
      color: "#111827",
      marginBottom: "8px",
    },
    fileMetadata: {
      fontSize: "12px",
      color: "#6b7280",
      lineHeight: 1.5,
    },
    visionToggle: {
      padding: "12px",
      backgroundColor: "#f8f9fa",
      borderRadius: "8px",
      marginBottom: "16px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    visionTitle: {
      fontSize: "16px",
      fontWeight: 600,
      color: "#374151",
    },
    toggleButton: {
      display: "flex",
      alignItems: "center",
      gap: "6px",
      padding: "6px 12px",
      backgroundColor: "white",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "14px",
    },
    visionResults: {
      padding: "12px",
      backgroundColor: "#e8f5e9",
      borderRadius: "8px",
      marginBottom: "16px",
    },
    pdfContainer: {
      position: "relative" as const,
      backgroundColor: "#f5f5f5",
      borderRadius: "8px",
      padding: "16px",
      minHeight: "600px",
      display: "flex",
      flexDirection: "column" as const,
      alignItems: "center",
      justifyContent: "center",
    },
    pdfMock: {
      width: "100%",
      maxWidth: "600px",
      height: "800px",
      backgroundColor: "white",
      border: "1px solid #d1d5db",
      borderRadius: "4px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "#9ca3af",
      fontSize: "24px",
    },
    pageControls: {
      marginTop: "16px",
      display: "flex",
      alignItems: "center",
      gap: "16px",
    },
    pageInfo: {
      fontSize: "14px",
      color: "#6b7280",
    },
    statusChip: {
      display: "inline-flex",
      alignItems: "center",
      padding: "4px 8px",
      borderRadius: "12px",
      fontSize: "12px",
      fontWeight: 500,
      marginLeft: "8px",
    },
    successChip: {
      backgroundColor: "#e8f5e9",
      color: "#2e7d32",
    },
    infoChip: {
      backgroundColor: "#e3f2fd",
      color: "#1565c0",
    },
    defaultChip: {
      backgroundColor: "#f5f5f5",
      color: "#616161",
    },
  };

  const getVisionStatus = (pdf: PdfMetadata | null) => {
    if (!pdf) return null;
    
    if (pdf.has_vision_results) {
      return (
        <span style={{ ...styles.statusChip, ...styles.successChip }}>
          {pdf.table_count} Tables Found
        </span>
      );
    }
    
    return (
      <span style={{ ...styles.statusChip, ...styles.defaultChip }}>
        Not Processed
      </span>
    );
  };

  return (
    <PageLayout title="S3 PDF Viewer">
      <div style={styles.container}>
        {/* Stats Header */}
        <div style={styles.statsHeader}>
          <div style={styles.statsInfo}>
            <FileText size={24} />
            <span>PDF Library ({pdfs.length} files, 4.8 MB)</span>
          </div>
          <button
            style={styles.toggleButton}
            onClick={handleRefresh}
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>

        {/* Filter Controls */}
        <div style={styles.filterContainer}>
          <div style={styles.filterRow}>
            <input
              type="text"
              placeholder="Search PDFs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={styles.searchInput}
            />
            <select
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              style={styles.selectInput}
            >
              <option value="all">All Sources</option>
              <option value="dropbox">Dropbox</option>
              <option value="google_drive">Google Drive</option>
              <option value="s3">S3</option>
            </select>
          </div>
        </div>

        {/* Main Content */}
        <div style={styles.mainContent}>
          {/* Left Panel - File Info */}
          <div style={styles.leftPanel}>
            <div style={styles.navigationHeader}>
              <span style={{ fontWeight: 600 }}>
                File {selectedFileIndex + 1} of {filteredPdfs.length}
              </span>
              <div style={styles.navButtons}>
                <button
                  style={{
                    ...styles.navButton,
                    opacity: selectedFileIndex === 0 ? 0.5 : 1,
                    cursor: selectedFileIndex === 0 ? "not-allowed" : "pointer",
                  }}
                  onClick={goToPrevious}
                  disabled={selectedFileIndex === 0}
                >
                  <ChevronLeft size={16} />
                </button>
                <button
                  style={{
                    ...styles.navButton,
                    opacity: selectedFileIndex === filteredPdfs.length - 1 ? 0.5 : 1,
                    cursor: selectedFileIndex === filteredPdfs.length - 1 ? "not-allowed" : "pointer",
                  }}
                  onClick={goToNext}
                  disabled={selectedFileIndex === filteredPdfs.length - 1}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>

            {selectedPdf && (
              <div style={styles.fileInfo}>
                <div style={styles.fileTitle}>
                  {selectedPdf.name}
                  {getVisionStatus(selectedPdf)}
                </div>
                <div style={styles.fileMetadata}>
                  <div>File Size: {(selectedPdf.size / 1024 / 1024).toFixed(2)} MB</div>
                  <div>Source: {selectedPdf.source}</div>
                  <div>Last Modified: {new Date(selectedPdf.synced_at).toLocaleDateString()}</div>
                  {selectedPdf.path && <div>Path: {selectedPdf.path}</div>}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - PDF Display */}
          <div style={styles.rightPanel}>
            {/* Vision Results Toggle */}
            <div style={styles.visionToggle}>
              <span style={styles.visionTitle}>Table Detection & Vision Processing</span>
              <button
                style={styles.toggleButton}
                onClick={() => setShowVisionResults(!showVisionResults)}
              >
                {showVisionResults ? <EyeOff size={16} /> : <Eye size={16} />}
                {showVisionResults ? "Hide" : "Show"} Vision Results
              </button>
            </div>

            {/* Vision Results */}
            {showVisionResults && selectedPdf?.has_vision_results && (
              <div style={styles.visionResults}>
                <div style={{ fontWeight: 600, marginBottom: "8px" }}>Vision Processing Results</div>
                <div style={{ fontSize: "14px", color: "#047857" }}>
                  ✅ {selectedPdf.table_count} tables detected and processed
                </div>
              </div>
            )}

            {/* PDF Container */}
            <div ref={pdfContainerRef} style={styles.pdfContainer}>
              <div style={styles.pdfMock}>
                PDF Preview
                <br />
                {selectedPdf?.name}
              </div>
              
              {/* Page Controls */}
              <div style={styles.pageControls}>
                <button
                  style={{
                    ...styles.navButton,
                    opacity: pageNumber === 1 ? 0.5 : 1,
                    cursor: pageNumber === 1 ? "not-allowed" : "pointer",
                  }}
                  onClick={goToPrevPage}
                  disabled={pageNumber === 1}
                >
                  <ChevronLeft size={16} />
                </button>
                <span style={styles.pageInfo}>
                  Page {pageNumber} of {numPages}
                </span>
                <button
                  style={{
                    ...styles.navButton,
                    opacity: pageNumber === numPages ? 0.5 : 1,
                    cursor: pageNumber === numPages ? "not-allowed" : "pointer",
                  }}
                  onClick={goToNextPage}
                  disabled={pageNumber === numPages}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  );
};

export default PdfViewer;