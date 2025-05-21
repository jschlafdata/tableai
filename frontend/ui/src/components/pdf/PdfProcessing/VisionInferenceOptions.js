import React, { useState } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Tooltip,
  Typography,
  TextField,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  IconButton,
  FormControlLabel,
  Switch,
  Paper,
  Divider,
  Chip,
  Card,
  CardContent,
  CardHeader
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SettingsIcon from '@mui/icons-material/Settings';
import CodeIcon from '@mui/icons-material/Code';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const VisionInferenceOptions = ({
  fileId,
  hasClassification,
  visionLoading,
  onRunVisionInference,
  defaultOptions = {
    model_choice: 'gpt-4-vision',
    temperature: 0,
    top_k: 40,
    top_p: 0.95,
    zoom: 1.0,
    max_attempts: 3,
    timeout: 60,
    prompt: 'Analyze this document page and extract all relevant information.'
  }
}) => {
  // State for the options
  const [options, setOptions] = useState(defaultOptions);
  const [expanded, setExpanded] = useState(false);
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [visionResponse, setVisionResponse] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);

  // Example response - replace this with your actual response handling
  const exampleResponse = [
    {
        "number_of_tables": 3,
        "headers_heirarchy": false,
        "columns": {
            "0": [
                "Process",
                "Number Sales",
                "Net Sales",
                "Adjustments",
                "Chargebacks",
                "Disc",
                "3rd Party",
                "Net Deposits"
            ],
            "1": [
                "Card Type",
                "Sales",
                "Amount of Sales",
                "Settled",
                "Amount of Credits",
                "Amount of Net Sales",
                "Average Ticket",
                "Settled Per Item",
                "Disc Rate",
                "Processing Fees"
            ],
            "2": [
                "Number",
                "Amount",
                "Description",
                "Rate",
                "Total"
            ]
        }
    }
  ];

  // Available model choices
  const modelChoices = [
    { value: 'mistralVL', label: 'mistral-small3.1:24b' },
  ];

  // Handle option changes
  const handleOptionChange = (field, value) => {
    setOptions(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Handle running inference with the current options
  const handleRunInference = () => {
    // Show loading state
    onRunVisionInference(options);
    
    // Simulate API response - replace with your actual API call
    setTimeout(() => {
      setVisionResponse(exampleResponse);
    }, 1500);
  };

  // Copy response to clipboard
  const handleCopyResponse = () => {
    if (visionResponse) {
      navigator.clipboard.writeText(JSON.stringify(visionResponse, null, 2));
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  // Download response as JSON file
  const handleDownloadResponse = () => {
    if (visionResponse) {
      const blob = new Blob([JSON.stringify(visionResponse, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'vision_inference_response.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Detect if the prompt might be JSON or Markdown
  const detectFormat = (text) => {
    if (text.trim().startsWith('{') && text.trim().endsWith('}')) {
      try {
        JSON.parse(text);
        return 'json';
      } catch (e) {
        // Not valid JSON
      }
    }
    
    // Check for markdown indicators (headings, lists, etc.)
    if (text.match(/^#+\s|^\*\s|-\s\[|^>\s|\*\*.+\*\*|__.+__/m)) {
      return 'markdown';
    }
    
    return 'plaintext';
  };

  // Get the prompt format
  const promptFormat = detectFormat(options.prompt);

  // Component for rendering the prompt in the appropriate format
  const PromptPreview = () => {
    switch (promptFormat) {
      case 'json':
        try {
          const formattedJson = JSON.stringify(JSON.parse(options.prompt), null, 2);
          return (
            <SyntaxHighlighter 
              language="json" 
              style={atomDark}
              customStyle={{ borderRadius: '4px', maxHeight: '400px' }}
            >
              {formattedJson}
            </SyntaxHighlighter>
          );
        } catch (e) {
          return <pre>{options.prompt}</pre>;
        }
      case 'markdown':
        return (
          <Box sx={{ 
            p: 2, 
            border: '1px solid rgba(0, 0, 0, 0.12)', 
            borderRadius: '4px',
            backgroundColor: '#f5f5f5',
            maxHeight: '400px',
            overflow: 'auto'
          }}>
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                // Custom component for code blocks to add syntax highlighting
                code: ({node, inline, className, children, ...props}) => {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      language={match[1]}
                      style={atomDark}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {options.prompt}
            </ReactMarkdown>
          </Box>
        );
      default:
        return (
          <Box sx={{ 
            p: 2, 
            border: '1px solid rgba(0, 0, 0, 0.12)', 
            borderRadius: '4px',
            backgroundColor: '#f5f5f5',
            maxHeight: '400px',
            overflow: 'auto',
            whiteSpace: 'pre-wrap'
          }}>
            <pre style={{ margin: 0 }}>{options.prompt}</pre>
          </Box>
        );
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Accordion 
        expanded={expanded} 
        onChange={() => setExpanded(!expanded)}
        sx={{ mb: 2, boxShadow: 'none', border: '1px solid rgba(0, 0, 0, 0.12)' }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="vision-options-content"
          id="vision-options-header"
          sx={{ 
            '&.Mui-expanded': { minHeight: 48 },
            '& .MuiAccordionSummary-content.Mui-expanded': { margin: '12px 0' }
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon fontSize="small" />
            <Typography>Vision Inference Options</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
            {/* Model Selection */}
            <TextField
              select
              label="Model"
              value={options.model_choice}
              onChange={(e) => handleOptionChange('model_choice', e.target.value)}
              fullWidth
              size="small"
              helperText="Select vision model"
            >
              {modelChoices.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </TextField>

            {/* Temperature */}
            <Box>
              <Typography variant="body2" gutterBottom>Temperature: {options.temperature}</Typography>
              <Slider
                value={options.temperature}
                min={0}
                max={1}
                step={0.1}
                onChange={(_, value) => handleOptionChange('temperature', value)}
                aria-labelledby="temperature-slider"
              />
            </Box>

            {/* Top K */}
            <TextField
              label="Top K"
              type="number"
              value={options.top_k}
              onChange={(e) => handleOptionChange('top_k', parseInt(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 1, max: 100 }}
            />

            {/* Top P */}
            <TextField
              label="Top P"
              type="number"
              value={options.top_p}
              onChange={(e) => handleOptionChange('top_p', parseFloat(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
            />

            {/* Zoom */}
            <TextField
              label="Zoom"
              type="number"
              value={options.zoom}
              onChange={(e) => handleOptionChange('zoom', parseFloat(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 0.1, max: 5, step: 0.1 }}
            />

            {/* Max Attempts */}
            <TextField
              label="Max Attempts"
              type="number"
              value={options.max_attempts}
              onChange={(e) => handleOptionChange('max_attempts', parseInt(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 1, max: 10 }}
            />

            {/* Timeout */}
            <TextField
              label="Timeout (seconds)"
              type="number"
              value={options.timeout}
              onChange={(e) => handleOptionChange('timeout', parseInt(e.target.value))}
              fullWidth
              size="small"
              inputProps={{ min: 10, max: 300 }}
            />

            <TextField
                label="Page Limit"
                type="number"
                value={options.page_limit}
                onChange={(e) => handleOptionChange('page_limit', parseInt(e.target.value))}
                fullWidth
                size="small"
                inputProps={{ min: 1, max: 100 }}
                helperText="Maximum number of pages to process"
            />
          </Box>

          {/* Prompt Section with Editor/Preview Toggle */}
          <Box sx={{ mt: 2, mb: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="body1">Prompt</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={isPreviewMode}
                      onChange={(e) => setIsPreviewMode(e.target.checked)}
                      size="small"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <CodeIcon fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="body2">Preview</Typography>
                    </Box>
                  }
                />
              </Box>
            </Box>

            {isPreviewMode ? (
              <PromptPreview />
            ) : (
              <TextField
                multiline
                minRows={3}
                maxRows={20}
                value={options.prompt}
                onChange={(e) => handleOptionChange('prompt', e.target.value)}
                fullWidth
                variant="outlined"
                size="small"
                InputProps={{
                  sx: {
                    height: 'auto',
                    fontFamily: 'monospace'
                  }
                }}
                placeholder="Enter your prompt here..."
              />
            )}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Run Vision Button */}
      <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
        <span>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleRunInference}
            disabled={visionLoading || !fileId || !hasClassification}
            startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
            fullWidth
          >
            {visionLoading ? "Running Vision Inference..." : "Run Vision Inference"}
          </Button>
        </span>
      </Tooltip>

      {/* Response Display Section */}
      {visionResponse && (
        <Card 
          variant="outlined" 
          sx={{ 
            mt: 3, 
            overflow: 'visible',
            borderRadius: 1,
            boxShadow: '0 2px 10px rgba(0,0,0,0.08)'
          }}
        >
          <CardHeader
            title={
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
                    Vision Inference Result
                  </Typography>
                  <Chip 
                    label={`${visionResponse[0].number_of_tables} tables detected`} 
                    size="small" 
                    color="primary" 
                    sx={{ ml: 2 }}
                  />
                </Box>
                <Box>
                  <Tooltip title="Toggle Color Theme">
                    <Switch
                      checked={isDarkMode}
                      onChange={(e) => setIsDarkMode(e.target.checked)}
                      size="small"
                    />
                  </Tooltip>
                  <Tooltip title={copySuccess ? "Copied!" : "Copy JSON"}>
                    <IconButton onClick={handleCopyResponse} size="small" sx={{ ml: 1 }}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Download JSON">
                    <IconButton onClick={handleDownloadResponse} size="small">
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
            }
            sx={{ 
              pb: 0,
              borderBottom: '1px solid rgba(0,0,0,0.08)'
            }}
          />
          <CardContent sx={{ p: 0 }}>
            <SyntaxHighlighter 
              language="json" 
              style={isDarkMode ? atomDark : oneLight}
              customStyle={{ 
                margin: 0, 
                borderRadius: '0 0 4px 4px', 
                maxHeight: '500px',
                fontSize: '0.9rem'
              }}
            >
              {JSON.stringify(visionResponse, null, 2)}
            </SyntaxHighlighter>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default VisionInferenceOptions;

// import React, { useState } from 'react';
// import {
//   Box,
//   Button,
//   CircularProgress,
//   Tooltip,
//   Typography,
//   TextField,
//   MenuItem,
//   Accordion,
//   AccordionSummary,
//   AccordionDetails,
//   Slider,
//   IconButton,
//   FormControlLabel,
//   Switch
// } from '@mui/material';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import SettingsIcon from '@mui/icons-material/Settings';
// import CodeIcon from '@mui/icons-material/Code';
// import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
// import ReactMarkdown from 'react-markdown';
// import remarkGfm from 'remark-gfm';

// const VisionInferenceOptions = ({
//   fileId,
//   hasClassification,
//   visionLoading,
//   onRunVisionInference,
//   defaultOptions = {
//     model_choice: 'gpt-4-vision',
//     temperature: 0,
//     top_k: 40,
//     top_p: 0.95,
//     zoom: 1.0,
//     max_attempts: 3,
//     timeout: 60,
//     prompt: 'Analyze this document page and extract all relevant information.'
//   }
// }) => {
//   // State for the options
//   const [options, setOptions] = useState(defaultOptions);
//   const [expanded, setExpanded] = useState(false);
//   const [isPreviewMode, setIsPreviewMode] = useState(false);

//   // Available model choices
//   const modelChoices = [
//     { value: 'mistralVL', label: 'mistral-small3.1:24b' },
//   ];

//   // Handle option changes
//   const handleOptionChange = (field, value) => {
//     setOptions(prev => ({
//       ...prev,
//       [field]: value
//     }));
//   };

//   // Handle running inference with the current options
//   const handleRunInference = () => {
//     onRunVisionInference(options);
//   };

//   // Detect if the prompt might be JSON or Markdown
//   const detectFormat = (text) => {
//     if (text.trim().startsWith('{') && text.trim().endsWith('}')) {
//       try {
//         JSON.parse(text);
//         return 'json';
//       } catch (e) {
//         // Not valid JSON
//       }
//     }
    
//     // Check for markdown indicators (headings, lists, etc.)
//     if (text.match(/^#+\s|^\*\s|-\s\[|^>\s|\*\*.+\*\*|__.+__/m)) {
//       return 'markdown';
//     }
    
//     return 'plaintext';
//   };

//   // Get the prompt format
//   const promptFormat = detectFormat(options.prompt);

//   // Component for rendering the prompt in the appropriate format
//   const PromptPreview = () => {
//     switch (promptFormat) {
//       case 'json':
//         try {
//           const formattedJson = JSON.stringify(JSON.parse(options.prompt), null, 2);
//           return (
//             <SyntaxHighlighter 
//               language="json" 
//               style={atomDark}
//               customStyle={{ borderRadius: '4px', maxHeight: '400px' }}
//             >
//               {formattedJson}
//             </SyntaxHighlighter>
//           );
//         } catch (e) {
//           return <pre>{options.prompt}</pre>;
//         }
//       case 'markdown':
//         return (
//           <Box sx={{ 
//             p: 2, 
//             border: '1px solid rgba(0, 0, 0, 0.12)', 
//             borderRadius: '4px',
//             backgroundColor: '#f5f5f5',
//             maxHeight: '400px',
//             overflow: 'auto'
//           }}>
//             <ReactMarkdown 
//               remarkPlugins={[remarkGfm]}
//               components={{
//                 // Custom component for code blocks to add syntax highlighting
//                 code: ({node, inline, className, children, ...props}) => {
//                   const match = /language-(\w+)/.exec(className || '');
//                   return !inline && match ? (
//                     <SyntaxHighlighter
//                       language={match[1]}
//                       style={atomDark}
//                       PreTag="div"
//                       {...props}
//                     >
//                       {String(children).replace(/\n$/, '')}
//                     </SyntaxHighlighter>
//                   ) : (
//                     <code className={className} {...props}>
//                       {children}
//                     </code>
//                   );
//                 }
//               }}
//             >
//               {options.prompt}
//             </ReactMarkdown>
//           </Box>
//         );
//       default:
//         return (
//           <Box sx={{ 
//             p: 2, 
//             border: '1px solid rgba(0, 0, 0, 0.12)', 
//             borderRadius: '4px',
//             backgroundColor: '#f5f5f5',
//             maxHeight: '400px',
//             overflow: 'auto',
//             whiteSpace: 'pre-wrap'
//           }}>
//             <pre style={{ margin: 0 }}>{options.prompt}</pre>
//           </Box>
//         );
//     }
//   };

//   return (
//     <Box sx={{ width: '100%' }}>
//       <Accordion 
//         expanded={expanded} 
//         onChange={() => setExpanded(!expanded)}
//         sx={{ mb: 2, boxShadow: 'none', border: '1px solid rgba(0, 0, 0, 0.12)' }}
//       >
//         <AccordionSummary
//           expandIcon={<ExpandMoreIcon />}
//           aria-controls="vision-options-content"
//           id="vision-options-header"
//           sx={{ 
//             '&.Mui-expanded': { minHeight: 48 },
//             '& .MuiAccordionSummary-content.Mui-expanded': { margin: '12px 0' }
//           }}
//         >
//           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
//             <SettingsIcon fontSize="small" />
//             <Typography>Vision Inference Options</Typography>
//           </Box>
//         </AccordionSummary>
//         <AccordionDetails>
//           <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
//             {/* Model Selection */}
//             <TextField
//               select
//               label="Model"
//               value={options.model_choice}
//               onChange={(e) => handleOptionChange('model_choice', e.target.value)}
//               fullWidth
//               size="small"
//               helperText="Select vision model"
//             >
//               {modelChoices.map((option) => (
//                 <MenuItem key={option.value} value={option.value}>
//                   {option.label}
//                 </MenuItem>
//               ))}
//             </TextField>

//             {/* Temperature */}
//             <Box>
//               <Typography variant="body2" gutterBottom>Temperature: {options.temperature}</Typography>
//               <Slider
//                 value={options.temperature}
//                 min={0}
//                 max={1}
//                 step={0.1}
//                 onChange={(_, value) => handleOptionChange('temperature', value)}
//                 aria-labelledby="temperature-slider"
//               />
//             </Box>

//             {/* Top K */}
//             <TextField
//               label="Top K"
//               type="number"
//               value={options.top_k}
//               onChange={(e) => handleOptionChange('top_k', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 1, max: 100 }}
//             />

//             {/* Top P */}
//             <TextField
//               label="Top P"
//               type="number"
//               value={options.top_p}
//               onChange={(e) => handleOptionChange('top_p', parseFloat(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 0, max: 1, step: 0.01 }}
//             />

//             {/* Zoom */}
//             <TextField
//               label="Zoom"
//               type="number"
//               value={options.zoom}
//               onChange={(e) => handleOptionChange('zoom', parseFloat(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 0.1, max: 5, step: 0.1 }}
//             />

//             {/* Max Attempts */}
//             <TextField
//               label="Max Attempts"
//               type="number"
//               value={options.max_attempts}
//               onChange={(e) => handleOptionChange('max_attempts', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 1, max: 10 }}
//             />

//             {/* Timeout */}
//             <TextField
//               label="Timeout (seconds)"
//               type="number"
//               value={options.timeout}
//               onChange={(e) => handleOptionChange('timeout', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 10, max: 300 }}
//             />

//             <TextField
//                 label="Page Limit"
//                 type="number"
//                 value={options.page_limit}
//                 onChange={(e) => handleOptionChange('page_limit', parseInt(e.target.value))}
//                 fullWidth
//                 size="small"
//                 inputProps={{ min: 1, max: 100 }}
//                 helperText="Maximum number of pages to process"
//             />
//           </Box>

//           {/* Prompt Section with Editor/Preview Toggle */}
//           <Box sx={{ mt: 2, mb: 1 }}>
//             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
//               <Typography variant="body1">Prompt</Typography>
//               <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                 <FormControlLabel
//                   control={
//                     <Switch
//                       checked={isPreviewMode}
//                       onChange={(e) => setIsPreviewMode(e.target.checked)}
//                       size="small"
//                     />
//                   }
//                   label={
//                     <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                       <CodeIcon fontSize="small" sx={{ mr: 0.5 }} />
//                       <Typography variant="body2">Preview</Typography>
//                     </Box>
//                   }
//                 />
//               </Box>
//             </Box>

//             {isPreviewMode ? (
//               <PromptPreview />
//             ) : (
//               <TextField
//                 multiline
//                 minRows={3}
//                 maxRows={20}
//                 value={options.prompt}
//                 onChange={(e) => handleOptionChange('prompt', e.target.value)}
//                 fullWidth
//                 variant="outlined"
//                 size="small"
//                 InputProps={{
//                   sx: {
//                     height: 'auto',
//                     fontFamily: 'monospace'
//                   }
//                 }}
//                 placeholder="Enter your prompt here..."
//               />
//             )}
//           </Box>
//         </AccordionDetails>
//       </Accordion>

//       {/* Run Vision Button */}
//       <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
//         <span>
//           <Button
//             variant="contained"
//             color="secondary"
//             onClick={handleRunInference}
//             disabled={visionLoading || !fileId || !hasClassification}
//             startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
//             fullWidth
//           >
//             {visionLoading ? "Running Vision Inference..." : "Run Vision Inference"}
//           </Button>
//         </span>
//       </Tooltip>
//     </Box>
//   );
// };

// export default VisionInferenceOptions;


// import React, { useState } from 'react';
// import {
//   Box,
//   Button,
//   CircularProgress,
//   Tooltip,
//   Typography,
//   TextField,
//   MenuItem,
//   Accordion,
//   AccordionSummary,
//   AccordionDetails,
//   Slider,
//   IconButton
// } from '@mui/material';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import SettingsIcon from '@mui/icons-material/Settings';

// const VisionInferenceOptions = ({
//   fileId,
//   hasClassification,
//   visionLoading,
//   onRunVisionInference,
//   defaultOptions = {
//     model_choice: 'gpt-4-vision',
//     temperature: 0,
//     top_k: 40,
//     top_p: 0.95,
//     zoom: 1.0,
//     max_attempts: 3,
//     timeout: 60,
//     prompt: 'Analyze this document page and extract all relevant information.'
//   }
// }) => {
//   // State for the options
//   const [options, setOptions] = useState(defaultOptions);
//   const [expanded, setExpanded] = useState(false);

//   // Available model choices
//   const modelChoices = [
//     { value: 'mistralVL', label: 'mistral-small3.1:24b' },
//   ];

//   // Handle option changes
//   const handleOptionChange = (field, value) => {
//     setOptions(prev => ({
//       ...prev,
//       [field]: value
//     }));
//   };

//   // Handle running inference with the current options
//   const handleRunInference = () => {
//     onRunVisionInference(options);
//   };

//   return (
//     <Box sx={{ width: '100%' }}>
//       <Accordion 
//         expanded={expanded} 
//         onChange={() => setExpanded(!expanded)}
//         sx={{ mb: 2, boxShadow: 'none', border: '1px solid rgba(0, 0, 0, 0.12)' }}
//       >
//         <AccordionSummary
//           expandIcon={<ExpandMoreIcon />}
//           aria-controls="vision-options-content"
//           id="vision-options-header"
//           sx={{ 
//             '&.Mui-expanded': { minHeight: 48 },
//             '& .MuiAccordionSummary-content.Mui-expanded': { margin: '12px 0' }
//           }}
//         >
//           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
//             <SettingsIcon fontSize="small" />
//             <Typography>Vision Inference Options</Typography>
//           </Box>
//         </AccordionSummary>
//         <AccordionDetails>
//           <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
//             {/* Model Selection */}
//             <TextField
//               select
//               label="Model"
//               value={options.model_choice}
//               onChange={(e) => handleOptionChange('model_choice', e.target.value)}
//               fullWidth
//               size="small"
//               helperText="Select vision model"
//             >
//               {modelChoices.map((option) => (
//                 <MenuItem key={option.value} value={option.value}>
//                   {option.label}
//                 </MenuItem>
//               ))}
//             </TextField>

//             {/* Temperature */}
//             <Box>
//               <Typography variant="body2" gutterBottom>Temperature: {options.temperature}</Typography>
//               <Slider
//                 value={options.temperature}
//                 min={0}
//                 max={1}
//                 step={0.1}
//                 onChange={(_, value) => handleOptionChange('temperature', value)}
//                 aria-labelledby="temperature-slider"
//               />
//             </Box>

//             {/* Top K */}
//             <TextField
//               label="Top K"
//               type="number"
//               value={options.top_k}
//               onChange={(e) => handleOptionChange('top_k', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 1, max: 100 }}
//             />

//             {/* Top P */}
//             <TextField
//               label="Top P"
//               type="number"
//               value={options.top_p}
//               onChange={(e) => handleOptionChange('top_p', parseFloat(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 0, max: 1, step: 0.01 }}
//             />

//             {/* Zoom */}
//             <TextField
//               label="Zoom"
//               type="number"
//               value={options.zoom}
//               onChange={(e) => handleOptionChange('zoom', parseFloat(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 0.1, max: 5, step: 0.1 }}
//             />

//             {/* Max Attempts */}
//             <TextField
//               label="Max Attempts"
//               type="number"
//               value={options.max_attempts}
//               onChange={(e) => handleOptionChange('max_attempts', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 1, max: 10 }}
//             />

//             {/* Timeout */}
//             <TextField
//               label="Timeout (seconds)"
//               type="number"
//               value={options.timeout}
//               onChange={(e) => handleOptionChange('timeout', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 10, max: 300 }}
//             />

//             <TextField
//                 label="Page Limit"
//                 type="number"
//                 value={options.page_limit}
//                 onChange={(e) => handleOptionChange('page_limit', parseInt(e.target.value))}
//                 fullWidth
//                 size="small"
//                 inputProps={{ min: 1, max: 100 }}
//                 helperText="Maximum number of pages to process"
//             />
//           </Box>

//           {/* Prompt - full width with auto-expanding height */}
//           <TextField
//             label="Prompt"
//             multiline
//             minRows={3}
//             maxRows={20}
//             value={options.prompt}
//             onChange={(e) => handleOptionChange('prompt', e.target.value)}
//             fullWidth
//             margin="normal"
//             variant="outlined"
//             size="small"
//             InputProps={{
//               sx: {
//                 height: 'auto'
//               }
//             }}
//           />
//         </AccordionDetails>
//       </Accordion>

//       {/* Run Vision Button */}
//       <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
//         <span>
//           <Button
//             variant="contained"
//             color="secondary"
//             onClick={handleRunInference}
//             disabled={visionLoading || !fileId || !hasClassification}
//             startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
//             fullWidth
//           >
//             {visionLoading ? "Running Vision Inference..." : "Run Vision Inference"}
//           </Button>
//         </span>
//       </Tooltip>
//     </Box>
//   );
// };

// export default VisionInferenceOptions;