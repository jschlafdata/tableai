// src/components/classify/ClassifyPage.tsx
import { useState } from "react";
import type { CSSProperties } from "react";
import {
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  Save,
} from "lucide-react";
import PageLayout from "../layouts/PageLayout";
import type { User } from "../../types/user";

// Types
interface ClassificationLabel {
  classification: string;
  label: string;
  isOriginal?: boolean;
}

interface ClassificationSample {
  classification: string;
  samples: string[];
}

interface ClassifyPageProps {
  user: User | null;
}

// Mock data for UI demonstration
const MOCK_CLASSIFICATIONS: ClassificationLabel[] = [
  { classification: "invoice", label: "Invoice Document", isOriginal: true },
  { classification: "receipt", label: "Receipt", isOriginal: true },
  { classification: "contract", label: "", isOriginal: false },
  { classification: "report", label: "", isOriginal: false },
];

const MOCK_SAMPLES: ClassificationSample[] = [
  { classification: "invoice", samples: ["sample1.pdf", "sample2.pdf", "sample3.pdf"] },
  { classification: "receipt", samples: ["receipt1.pdf", "receipt2.pdf"] },
  { classification: "contract", samples: ["contract1.pdf", "contract2.pdf", "contract3.pdf", "contract4.pdf"] },
  { classification: "report", samples: ["report1.pdf"] },
];

const ClassifyPage: React.FC<ClassifyPageProps> = ({ user }) => {
  // State management
  const [labels, setLabels] = useState<Record<string, string>>(
    MOCK_CLASSIFICATIONS.reduce((acc, item) => ({
      ...acc,
      [item.classification]: item.label,
    }), {})
  );
  
  const [originalLabels] = useState<Record<string, boolean>>(
    MOCK_CLASSIFICATIONS.reduce((acc, item) => ({
      ...acc,
      [item.classification]: item.isOriginal || false,
    }), {})
  );

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [pageIndices, setPageIndices] = useState<Record<string, number>>({});
  const [savingStatus, setSavingStatus] = useState<Record<string, boolean>>({});
  const [savedMessages, setSavedMessages] = useState<Record<string, string>>({});

  // Convert samples array to object for easier access
  const samplesMap = MOCK_SAMPLES.reduce((acc, item) => ({
    ...acc,
    [item.classification]: item.samples,
  }), {} as Record<string, string[]>);

  // Handlers
  const handleLabelChange = (classification: string, value: string) => {
    setLabels(prev => ({ ...prev, [classification]: value }));
  };

  const handleToggleExpand = (classification: string) => {
    setExpanded(prev => ({ ...prev, [classification]: !prev[classification] }));
  };

  const handleNextSample = (classification: string) => {
    const samples = samplesMap[classification] || [];
    setPageIndices(prev => ({
      ...prev,
      [classification]: Math.min((prev[classification] || 0) + 1, samples.length - 1)
    }));
  };

  const handlePrevSample = (classification: string) => {
    setPageIndices(prev => ({
      ...prev,
      [classification]: Math.max((prev[classification] || 0) - 1, 0)
    }));
  };

  const handleSaveLabel = (classification: string) => {
    setSavingStatus(prev => ({ ...prev, [classification]: true }));
    
    // Simulate save operation
    setTimeout(() => {
      setSavingStatus(prev => ({ ...prev, [classification]: false }));
      setSavedMessages(prev => ({ 
        ...prev, 
        [classification]: "Saved successfully" 
      }));
      
      setTimeout(() => {
        setSavedMessages(prev => ({ ...prev, [classification]: "" }));
      }, 3000);
    }, 1000);
  };

  const handleSaveAll = () => {
    const allClassifications = Object.keys(labels);
    allClassifications.forEach(classification => {
      setSavingStatus(prev => ({ ...prev, [classification]: true }));
    });
    
    // Simulate batch save
    setTimeout(() => {
      allClassifications.forEach(classification => {
        setSavingStatus(prev => ({ ...prev, [classification]: false }));
        setSavedMessages(prev => ({ 
          ...prev, 
          [classification]: "Saved as part of batch update" 
        }));
      });
      
      setTimeout(() => {
        setSavedMessages({});
      }, 3000);
    }, 1500);
  };

  const styles: Record<string, CSSProperties> = {
    container: {
      padding: "24px",
    },
    title: {
      fontSize: "1.5rem",
      fontWeight: 600,
      color: "#111827",
      marginBottom: "24px",
    },
    subtitle: {
      fontSize: "1.125rem",
      fontWeight: 600,
      marginBottom: "16px",
      color: "#374151",
    },
    saveAllContainer: {
      display: "flex",
      justifyContent: "flex-end",
      marginBottom: "16px",
    },
    saveAllButton: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 16px",
      backgroundColor: "#2563eb",
      color: "white",
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "14px",
      fontWeight: 500,
    },
    paper: {
      padding: "16px",
      marginBottom: "16px",
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      position: "relative" as const,
      maxWidth: "750px",
    },
    paperClassified: {
      padding: "16px",
      marginBottom: "16px",
      backgroundColor: "white",
      borderRadius: "8px",
      border: "1px solid #4caf50",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      position: "relative" as const,
      maxWidth: "750px",
    },
    classifiedChip: {
      position: "absolute" as const,
      top: "8px",
      right: "40px",
      display: "flex",
      alignItems: "center",
      gap: "4px",
      padding: "4px 8px",
      backgroundColor: "#e8f5e9",
      color: "#2e7d32",
      borderRadius: "12px",
      fontSize: "12px",
      fontWeight: 500,
    },
    classificationHeader: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      maxWidth: "600px",
      marginBottom: "12px",
    },
    classificationText: {
      flex: 1,
      wordBreak: "break-word" as const,
      fontSize: "14px",
      color: "#374151",
    },
    expandButton: {
      padding: "4px",
      backgroundColor: "transparent",
      border: "none",
      cursor: "pointer",
      borderRadius: "4px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    inputContainer: {
      display: "flex",
      gap: "8px",
      marginTop: "8px",
      width: "600px",
      maxWidth: "100%",
    },
    textField: {
      flex: 1,
      padding: "8px 12px",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      fontSize: "14px",
      outline: "none",
    },
    textFieldClassified: {
      flex: 1,
      padding: "8px 12px",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      fontSize: "14px",
      outline: "none",
      backgroundColor: "rgba(76, 175, 80, 0.08)",
    },
    saveButton: {
      minWidth: "85px",
      padding: "8px 16px",
      backgroundColor: "white",
      color: "#2563eb",
      border: "1px solid #2563eb",
      borderRadius: "6px",
      cursor: "pointer",
      fontSize: "14px",
      display: "flex",
      alignItems: "center",
      gap: "4px",
    },
    savedMessage: {
      marginTop: "8px",
      color: "#059669",
      fontSize: "12px",
      width: "600px",
      maxWidth: "100%",
    },
    previewContainer: {
      marginTop: "16px",
      textAlign: "left" as const,
    },
    previewImage: {
      maxWidth: "600px",
      maxHeight: "800px",
      border: "1px solid #e5e7eb",
      borderRadius: "4px",
    },
    paginationContainer: {
      display: "flex",
      alignItems: "center",
      justifyContent: "flex-start",
      marginTop: "8px",
      gap: "8px",
    },
    paginationButton: {
      padding: "4px",
      backgroundColor: "transparent",
      border: "1px solid #d1d5db",
      borderRadius: "4px",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    paginationText: {
      padding: "0 16px",
      fontSize: "14px",
      color: "#6b7280",
    },
  };

  return (
    <PageLayout title="Classifications" user={user}>
      <div style={styles.container}>
        <h2 style={styles.title}>Classify PDFs and Assign Labels</h2>

        <h3 style={styles.subtitle}>Assign Labels:</h3>
        
        <div style={styles.saveAllContainer}>
          <button
            style={{
              ...styles.saveAllButton,
              opacity: Object.values(savingStatus).some(status => status) ? 0.5 : 1,
              cursor: Object.values(savingStatus).some(status => status) ? "not-allowed" : "pointer",
            }}
            onClick={handleSaveAll}
            disabled={Object.values(savingStatus).some(status => status)}
          >
            <Save size={16} />
            Save All Labels
          </button>
        </div>

        {Object.keys(samplesMap).map((classification) => {
          const label = labels[classification] || "";
          const isOriginal = originalLabels[classification];
          const sampleList = samplesMap[classification] || [];
          const currentIndex = pageIndices[classification] || 0;
          const currentSample = sampleList[currentIndex];
          const isSaving = savingStatus[classification];
          const savedMessage = savedMessages[classification];

          return (
            <div
              key={classification}
              style={isOriginal ? styles.paperClassified : styles.paper}
            >
              {isOriginal && (
                <div style={styles.classifiedChip}>
                  <CheckCircle size={14} />
                  Classified
                </div>
              )}

              <div style={styles.classificationHeader}>
                <div
                  style={{
                    ...styles.classificationText,
                    fontWeight: isOriginal ? "bold" : "normal",
                  }}
                >
                  {classification}
                </div>
                <button
                  style={styles.expandButton}
                  onClick={() => handleToggleExpand(classification)}
                >
                  {expanded[classification] ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </button>
              </div>

              <div style={styles.inputContainer}>
                <input
                  type="text"
                  placeholder="Label"
                  value={label}
                  onChange={(e) => handleLabelChange(classification, e.target.value)}
                  style={isOriginal ? styles.textFieldClassified : styles.textField}
                />
                
                <button
                  style={{
                    ...styles.saveButton,
                    opacity: isSaving ? 0.5 : 1,
                    cursor: isSaving ? "not-allowed" : "pointer",
                  }}
                  onClick={() => handleSaveLabel(classification)}
                  disabled={isSaving}
                >
                  <Save size={14} />
                  {isSaving ? "Saving..." : "Save"}
                </button>
              </div>

              {savedMessage && (
                <div style={styles.savedMessage}>
                  ✅ {savedMessage}
                </div>
              )}

              {expanded[classification] && currentSample && (
                <div style={styles.previewContainer}>
                  <div
                    style={{
                      ...styles.previewImage,
                      height: "400px",
                      backgroundColor: "#f3f4f6",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#6b7280",
                    }}
                  >
                    PDF Preview: {currentSample}
                  </div>
                  
                  <div style={styles.paginationContainer}>
                    <button
                      style={{
                        ...styles.paginationButton,
                        opacity: currentIndex === 0 ? 0.5 : 1,
                        cursor: currentIndex === 0 ? "not-allowed" : "pointer",
                      }}
                      onClick={() => handlePrevSample(classification)}
                      disabled={currentIndex === 0}
                    >
                      <ChevronLeft size={20} />
                    </button>
                    
                    <span style={styles.paginationText}>
                      {currentIndex + 1} / {sampleList.length}
                    </span>
                    
                    <button
                      style={{
                        ...styles.paginationButton,
                        opacity: currentIndex === sampleList.length - 1 ? 0.5 : 1,
                        cursor: currentIndex === sampleList.length - 1 ? "not-allowed" : "pointer",
                      }}
                      onClick={() => handleNextSample(classification)}
                      disabled={currentIndex === sampleList.length - 1}
                    >
                      <ChevronRight size={20} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </PageLayout>
  );
};

export default ClassifyPage;