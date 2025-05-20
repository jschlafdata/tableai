import React, { useState } from 'react';
import { Box, IconButton, Collapse, Grid } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const FileMetadataDisplay = ({ metadata }) => {
  const [expanded, setExpanded] = useState(false);
  if (!metadata) return null;

  // Log to help diagnose any issues
  console.log("CURRENT METADATA IN DISPLAY COMPONENT:", metadata);
  
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  const getPathCategoryTags = () => {
    if (!metadata.path_categories) return [];
    return Object.entries(metadata.path_categories)
      .filter(([_, value]) => value && value !== '')
      .map(([key, value]) => `${key}: ${value}`);
  };

  // Get the correct file ID to display - prioritize file_id, then dropbox_id
  const displayFileId = metadata.file_id || metadata.dropbox_id || 'Unknown ID';

  // Tag colors for different metadata groups
  const tagColors = {
    fileName: '#e3f2fd',     // Light blue
    classification: '#ede7f6', // Light purple
    modified: '#e8f5e9',     // Light green
    size: '#fff8e1',         // Light amber
    directories: '#f3e5f5',  // Light purple
    categories: '#e0f7fa',   // Light cyan
    stages: '#fff3e0'        // Light orange
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'left' }}>
      <Box
        sx={{
          display: 'inline-block',
          border: '1px solid #ccc',
          borderRadius: '6px',
          p: 2,
          bgcolor: '#ffffff',
          width: 'auto',
          maxWidth: expanded ? '700px' : 'fit-content',
          minWidth: expanded ? '500px' : 'auto'
        }}
      >
        <div style={styles.fileInfoHeader}>
          <div style={{ ...styles.headerBanner, backgroundColor: tagColors.fileName }}>
          <span style={styles.headerBannerText}>
            {metadata.file_name || metadata.name || 'Unknown file'}
            {displayFileId ? `: [${displayFileId}]` : ''}
          </span>
          </div>
          <div style={styles.expandButtonWrapper}>
            <IconButton size="small" onClick={() => setExpanded((prev) => !prev)}>
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </div>
        </div>

        <Collapse in={expanded}>
          <Box>
            {/* Modified and Size in the same row */}
            <Grid container spacing={2} sx={{ mb: 0 }}>
              <Grid item xs={6}>
                <MetadataGroup 
                  label="Modified" 
                  tags={[formatDate(metadata.client_modified || metadata.server_modified)]} 
                  tagColor={tagColors.modified}
                />
              </Grid>
              <Grid item xs={6}>
                <MetadataGroup 
                  label="Size" 
                  tags={[metadata.size ? `${(metadata.size / 1024).toFixed(2)} KB` : 'N/A']} 
                  tagColor={tagColors.size}
                />
              </Grid>
            </Grid>

            {/* Group: Directories */}
            <MetadataGroup
              label="Directories"
              tags={metadata.directories?.length ? metadata.directories : ['None']}
              tagColor={tagColors.directories}
            />

            {/* Group: Categories */}
            <MetadataGroup 
              label="Categories" 
              tags={getPathCategoryTags().length ? getPathCategoryTags() : ['None']} 
              tagColor={tagColors.categories}
            />

            <Grid container spacing={2} sx={{ mb: 0 }}>
              <Grid item xs={6}>
                <MetadataGroup
                  label="Classification"
                  tags={metadata.classification ? [metadata.classification] : ['N/A']}
                  tagColor={tagColors.classification}
                />
              </Grid>
              <Grid item xs={6}>
                <MetadataGroup
                  label="Completed Stages"
                  tags={metadata.completed_stages?.length ? metadata.completed_stages.map(String) : ['0']}
                  tagColor={tagColors.stages}
                />
              </Grid>
            </Grid>
          </Box>
        </Collapse>
      </Box>
    </div>
  );
};

const MetadataGroup = ({ label, tags, tagColor = '#e3f2fd' }) => (
  <Box mb={2} p={2} border="1px solid #ddd" borderRadius="4px" bgcolor="#f9f9f9">
    <div style={styles.metadataRow}>
      <span style={styles.label}>{label}:</span>
      <div style={styles.tagGroup}>
        {tags.map((tag, index) => (
          <span key={`${tag}-${index}`} style={{ ...styles.tag, backgroundColor: tagColor }}>
            {tag}
          </span>
        ))}
      </div>
    </div>
  </Box>
);

const styles = {
  headerBanner: {
    width: '100%',
    minHeight: '40px',
    display: 'flex',
    alignItems: 'center',
    padding: '0 18px',
    borderRadius: '5px 5px 5px 5px',
    boxSizing: 'border-box',
    marginBottom: '6px',
  },
  headerBannerText: {
    fontSize: '1.00rem',
    fontWeight: 700,
    color: '#13325a',
    letterSpacing: '0.5px',
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  fileInfoHeader: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: '0px',
    position: 'relative'
  },
  expandButtonWrapper: {
    position: 'absolute',
    top: 0,
    right: 0
  },
  metadataRow: {
    display: 'flex',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    gap: '8px'
  },
  tagGroup: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px'
  },
  label: {
    fontWeight: 'bold',
    fontSize: '13px',
    marginRight: '6px',
    paddingTop: '4px'
  },
  tag: {
    padding: '3px 8px',
    borderRadius: '4px',
    fontSize: '13px',
    fontWeight: 'bold',
    whiteSpace: 'nowrap'
  }
};

export default FileMetadataDisplay;


// import React, { useState } from 'react';
// import { Box, IconButton, Collapse, Grid } from '@mui/material';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';

// const FileMetadataDisplay = ({ metadata }) => {
//   const [expanded, setExpanded] = useState(false);
//   if (!metadata) return null;

//   const formatDate = (dateString) => {
//     if (!dateString) return 'N/A';
//     try {
//       const date = new Date(dateString);
//       return date.toLocaleString();
//     } catch {
//       return dateString;
//     }
//   };

//   const getPathCategoryTags = () => {
//     if (!metadata.path_categories) return [];
//     return Object.entries(metadata.path_categories)
//       .filter(([_, value]) => value && value !== '')
//       .map(([key, value]) => `${key}: ${value}`);
//   };

//   console.log("CURRENT METADATA: ", metadata)

//   // Tag colors for different metadata groups (feel free to adjust)
//   const tagColors = {
//     fileName: '#e3f2fd',     // Light blue
//     classification: '#ede7f6', // Light purple
//     modified: '#e8f5e9',     // Light green
//     size: '#fff8e1',         // Light amber
//     directories: '#f3e5f5',  // Light purple
//     categories: '#e0f7fa',   // Light cyan
//     stages: '#fff3e0'        // Light orange
//   };

//   return (
//     <div style={{ display: 'flex', justifyContent: 'left' }}>
//       <Box
//         sx={{
//           display: 'inline-block',
//           border: '1px solid #ccc',
//           borderRadius: '6px',
//           p: 2,
//           bgcolor: '#ffffff',
//           width: 'auto',
//           maxWidth: expanded ? '700px' : 'fit-content',
//           minWidth: expanded ? '500px' : 'auto'
//         }}
//       >
//         <div style={styles.fileInfoHeader}>
//           <div style={{ ...styles.headerBanner, backgroundColor: tagColors.fileName }}>
//           <span style={styles.headerBannerText}>
//             {metadata.file_name}
//             {metadata.file_id ? `: [${metadata.file_id}]` : ''}
//           </span>
//           </div>
//           <div style={styles.expandButtonWrapper}>
//             <IconButton size="small" onClick={() => setExpanded((prev) => !prev)}>
//               {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//             </IconButton>
//           </div>
//         </div>

//         <Collapse in={expanded}>
//           <Box>
//             {/* Modified and Size in the same row */}
//             <Grid container spacing={2} sx={{ mb: 0 }}>
//               <Grid item xs={6}>
//                 <MetadataGroup 
//                   label="Modified" 
//                   tags={[formatDate(metadata.client_modified || metadata.server_modified)]} 
//                   tagColor={tagColors.modified}
//                 />
//               </Grid>
//               <Grid item xs={6}>
//                 <MetadataGroup 
//                   label="Size" 
//                   tags={[metadata.size ? `${(metadata.size / 1024).toFixed(2)} KB` : 'N/A']} 
//                   tagColor={tagColors.size}
//                 />
//               </Grid>
//             </Grid>

//             {/* Group: Directories */}
//             <MetadataGroup
//               label="Directories"
//               tags={metadata.directories?.length ? metadata.directories : ['None']}
//               tagColor={tagColors.directories}
//             />

//             {/* Group: Categories */}
//             <MetadataGroup 
//               label="Categories" 
//               tags={getPathCategoryTags().length ? getPathCategoryTags() : ['None']} 
//               tagColor={tagColors.categories}
//             />

//             <Grid container spacing={2} sx={{ mb: 0 }}>
//               <Grid item xs={6}>
//                 <MetadataGroup
//                   label="Classification"
//                   tags={metadata.classification ? [metadata.classification] : ['N/A']}
//                   tagColor={tagColors.classification}
//                 />
//               </Grid>
//               <Grid item xs={6}>
//                 <MetadataGroup
//                   label="Completed Stages"
//                   tags={metadata.completed_stages?.length ? metadata.completed_stages : ['0']}
//                   tagColor={tagColors.stages}
//                 />
//               </Grid>
//             </Grid>
//           </Box>
//         </Collapse>
//       </Box>
//     </div>
//   );
// };

// const MetadataGroup = ({ label, tags, tagColor = '#e3f2fd' }) => (
//   <Box mb={2} p={2} border="1px solid #ddd" borderRadius="4px" bgcolor="#f9f9f9">
//     <div style={styles.metadataRow}>
//       <span style={styles.label}>{label}:</span>
//       <div style={styles.tagGroup}>
//         {tags.map((tag) => (
//           <span key={tag} style={{ ...styles.tag, backgroundColor: tagColor }}>
//             {tag}
//           </span>
//         ))}
//       </div>
//     </div>
//   </Box>
// );

// const styles = {
//   headerBanner: {
//     width: '100%',
//     minHeight: '40px',
//     display: 'flex',
//     alignItems: 'center',
//     padding: '0 18px',
//     borderRadius: '5px 5px 5px 5px',
//     boxSizing: 'border-box',
//     marginBottom: '6px',
//   },
//   headerBannerText: {
//     fontSize: '1.00rem',
//     fontWeight: 700,
//     color: '#13325a',
//     letterSpacing: '0.5px',
//     flex: 1,
//     overflow: 'hidden',
//     textOverflow: 'ellipsis',
//     whiteSpace: 'nowrap',
//   },
//   fileInfoHeader: {
//     display: 'flex',
//     flexDirection: 'column',
//     alignItems: 'center',
//     marginBottom: '0px',
//     position: 'relative'
//   },
//   expandButtonWrapper: {
//     position: 'absolute',
//     top: 0,
//     right: 0
//   },
//   metadataRow: {
//     display: 'flex',
//     alignItems: 'flex-start',
//     flexWrap: 'wrap',
//     gap: '8px'
//   },
//   tagGroup: {
//     display: 'flex',
//     flexWrap: 'wrap',
//     gap: '8px'
//   },
//   label: {
//     fontWeight: 'bold',
//     fontSize: '13px',
//     marginRight: '6px',
//     paddingTop: '4px'
//   },
//   tag: {
//     padding: '3px 8px',
//     borderRadius: '4px',
//     fontSize: '13px',
//     fontWeight: 'bold',
//     whiteSpace: 'nowrap'
//   }
// };

// export default FileMetadataDisplay;

// // import React, { useState } from 'react';
// // import { Box, IconButton, Collapse, Grid } from '@mui/material';
// // import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// // import ExpandLessIcon from '@mui/icons-material/ExpandLess';

// // const FileMetadataDisplay = ({ metadata }) => {
// //   const [expanded, setExpanded] = useState(false);
// //   if (!metadata) return null;

// //   const formatDate = (dateString) => {
// //     if (!dateString) return 'N/A';
// //     try {
// //       const date = new Date(dateString);
// //       return date.toLocaleString();
// //     } catch {
// //       return dateString;
// //     }
// //   };

// //   const getPathCategoryTags = () => {
// //     if (!metadata.path_categories) return [];
// //     return Object.entries(metadata.path_categories)
// //       .filter(([_, value]) => value && value !== '')
// //       .map(([key, value]) => `${key}: ${value}`);
// //   };

// //   const renderTag = (text, color = '#e3f2fd', customStyle = {}) => (
// //     <span key={text} style={{ ...styles.tag, ...customStyle, backgroundColor: color }}>
// //       {text}
// //     </span>
// //   );

// //   // Define tag colors for different metadata groups
// //   const tagColors = {
// //     fileName: '#e3f2fd',     // Light blue
// //     classification: '#ede7f6', // Light purple
// //     modified: '#e8f5e9',     // Light green
// //     size: '#fff8e1',         // Light amber
// //     directories: '#f3e5f5',  // Light purple
// //     categories: '#e0f7fa',   // Light cyan
// //     stages: '#fff3e0'        // Light orange
// //   };

// //   return (
// //     <div style={{ display: 'flex', justifyContent: 'left' }}>
// //       <Box
// //         sx={{
// //           display: 'inline-block',
// //           border: '1px solid #ccc',
// //           borderRadius: '6px',
// //           p: 2,
// //           bgcolor: '#ffffff',
// //           width: 'auto',
// //           maxWidth: expanded ? '700px' : 'fit-content',
// //           minWidth: expanded ? '500px' : 'auto'
// //         }}
// //       >
// //     <div style={styles.fileInfoHeader}>
// //         <div style={{ ...styles.headerBanner, backgroundColor: tagColors.fileName }}>
// //             <span style={styles.headerBannerText}>
// //             {(metadata.file_name || metadata.name || '').replace(/\.[^/.]+$/, '')}
// //             </span>
// //         </div>
// //         <div style={styles.expandButtonWrapper}>
// //             <IconButton size="small" onClick={() => setExpanded((prev) => !prev)}>
// //             {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
// //             </IconButton>
// //         </div>
// //         </div>

// //         <Collapse in={expanded}>
// //           <Box>
// //             {/* Modified and Size in the same row */}
// //             <Grid container spacing={2} sx={{ mb: 2 }}>
// //               <Grid item xs={6}>
// //                 <MetadataGroup 
// //                   label="Modified" 
// //                   tags={[formatDate(metadata.client_modified || metadata.server_modified)]} 
// //                   tagColor={tagColors.modified}
// //                 />
// //               </Grid>
// //               <Grid item xs={6}>
// //                 <MetadataGroup 
// //                   label="Size" 
// //                   tags={[metadata.size ? `${(metadata.size / 1024).toFixed(2)} KB` : 'N/A']} 
// //                   tagColor={tagColors.size}
// //                 />
// //               </Grid>
// //             </Grid>

// //             {/* Group: Directories */}
// //             <MetadataGroup
// //               label="Directories"
// //               tags={metadata.directories?.length ? metadata.directories : ['None']}
// //               tagColor={tagColors.directories}
// //             />

// //             {/* Group: Categories */}
// //             <MetadataGroup 
// //               label="Categories" 
// //               tags={getPathCategoryTags().length ? getPathCategoryTags() : ['None']} 
// //               tagColor={tagColors.categories}
// //             />

// //             <Grid container spacing={2} sx={{ mb: 2 }}>
// //               <Grid item xs={6}>
// //                 <MetadataGroup
// //                     label="Classification"
// //                     tags={metadata.classification ? [metadata.classification] : ['N/A']}
// //                     tagColor={tagColors.classification}
// //                     />

// //               </Grid>
// //               <Grid item xs={6}>
// //                 <MetadataGroup
// //                     label="Completed Stages"
// //                     tags={metadata.completed_stages?.length ? metadata.completed_stages : ['0']}
// //                     tagColor={tagColors.stages}
// //                     />
// //               </Grid>
// //             </Grid>
// //           </Box>
// //         </Collapse>
// //       </Box>
// //     </div>
// //   );
// // };

// // const MetadataGroup = ({ label, tags, tagColor = '#e3f2fd' }) => (
// //   <Box mb={2} p={2} border="1px solid #ddd" borderRadius="4px" bgcolor="#f9f9f9">
// //     <div style={styles.metadataRow}>
// //       <span style={styles.label}>{label}:</span>
// //       <div style={styles.tagGroup}>
// //         {tags.map((tag) => (
// //           <span key={tag} style={{ ...styles.tag, backgroundColor: tagColor }}>
// //             {tag}
// //           </span>
// //         ))}
// //       </div>
// //     </div>
// //   </Box>
// // );

// // const styles = {
// //     headerBanner: {
// //         width: '100%',
// //         minHeight: '40px',
// //         display: 'flex',
// //         alignItems: 'center',
// //         padding: '0 18px',
// //         borderRadius: '5px 5px 0 0',
// //         boxSizing: 'border-box',
// //         marginBottom: '6px',
// //     },
// //     headerBannerText: {
// //         fontSize: '1.25rem',
// //         fontWeight: 700,
// //         color: '#13325a', // or any color that pops against your tagColors.fileName
// //         letterSpacing: '0.5px',
// //         flex: 1,
// //         overflow: 'hidden',
// //         textOverflow: 'ellipsis',
// //         whiteSpace: 'nowrap',
// //     },
// //   fileInfoHeader: {
// //     display: 'flex',
// //     flexDirection: 'column',
// //     alignItems: 'center',
// //     marginBottom: '0px',
// //     position: 'relative'
// //   },
// //   headerTag: {
// //     fontSize: '1.2rem',       // larger text
// //     fontWeight: 700,          // bolder
// //     padding: '6px 16px',      // more padding for emphasis
// //   },
// //   headerTagGroup: {
// //     display: 'flex',
// //     justifyContent: 'left',
// //     alignItems: 'left',
// //     gap: '10px',
// //     flexWrap: 'wrap',
// //     marginBottom: '4px'
// //   },
// //   expandButtonWrapper: {
// //     position: 'absolute',
// //     top: 0,
// //     right: 0
// //   },
// //   metadataRow: {
// //     display: 'flex',
// //     alignItems: 'flex-start',
// //     flexWrap: 'wrap',
// //     gap: '8px'
// //   },
// //   tagGroup: {
// //     display: 'flex',
// //     flexWrap: 'wrap',
// //     gap: '8px'
// //   },
// //   label: {
// //     fontWeight: 'bold',
// //     fontSize: '13px',
// //     marginRight: '6px',
// //     paddingTop: '4px'
// //   },
// //   tag: {
// //     padding: '3px 8px',
// //     borderRadius: '4px',
// //     fontSize: '13px',
// //     fontWeight: 'bold',
// //     whiteSpace: 'nowrap'
// //   }
// // };

// // export default FileMetadataDisplay;
