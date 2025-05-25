// Enhanced VisionInferenceOptions.jsx with Multi-Prompt Support
import React, { useState, useEffect } from 'react';
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
  CardHeader,
  Tabs,
  Tab,
  Badge,
  Menu,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SettingsIcon from '@mui/icons-material/Settings';
import CodeIcon from '@mui/icons-material/Code';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PlaylistPlayIcon from '@mui/icons-material/PlaylistPlay';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Table Structure Instruction Prompt
export const TABLE_STRUCTURE_PROMPT = `
Count how many complete, visually distinct tables appear in the image.
- Output this count as "number_of_tables".
- If no tables are identified, output "number_of_tables": 0.

For each table, determine whether it has hierarchical headers (multi-level columns).
- A table has hierarchical headers if there is at least one "parent" column spanning multiple sub-columns.
- Set "headers_hierarchy": true if the table has multi-level headers, otherwise false.

Return the columns of each table as the lowest-level column names in a flat list.

Include each table's details under a "tables" key in the output, using the table index (as a string) as the key.

Example Output:

{
  "number_of_tables": 2,
  "tables": {
    "0": {
      "headers_hierarchy": true,
      "columns": ["Col 1", "Col 2", "Col 3", "Col 4"]
    },
    "1": {
      "headers_hierarchy": false,
      "columns": ["Col X", "Col Y", "Col Z"]
    }
  }
}
`;

// Table Name/Total Row Prompt
export const TABLE_NAME_TOTALS_PROMPT = `
Given an image of a PDF page, identify all complete, visually distinct tables.

For each table, extract the following information:
1. Table Name: The title or label of the table if present. If no explicit name is found, use null.
2. Columns: A flat list of the lowest-level column names.
3. Totals Row Label: If the table includes a totals, summary, or grand total row at the bottom, return the text label or indicator of that row (e.g., "Monthly Total", "Grand Total", "Total", "Summary"). If no such row is present, return null. If the total row label is multi-line, be sure to include the entire value.
4. Bottom-Right Cell Value: Return the value found in the bottom-most right cell of the table (the last column of the last row), as a string. If not available, return null.
5. Table Breakers: Return a list of any sub-headers or sub-titles that appear within the table and would break or interrupt the table's structure (e.g., section headings, mid-table subtitles). If none are found, return an empty list.

Include each table’s details under a "tables" key in the output, using the table index (as a string) as the key.

Output the number of detected tables as "number_of_tables".

Example Output:

{
  "number_of_tables": 2,
  "tables": {
    "0": {
      "table_name": "Sales Summary",
      "columns": ["Product", "Quantity", "Unit Price", "Total"],
      "totals_row_label": "Monthly Total",
      "bottom_right_cell_value": "67,279.32",
      "table_breakers": []
    },
    "1": {
      "table_name": null,
      "columns": ["Date", "Description", "Amount"],
      "totals_row_label": null,
      "bottom_right_cell_value": "91.30",
      "table_breakers": ["Returned Items", "Adjustments & Credits"]
    }
  }
}
`;

const VisionInferenceOptions = ({
  fileId,
  hasClassification,
  visionLoading,
  onRunVisionInference,
  visionResponse = null,
  defaultOptions = {
    model_choice: 'mistralVL',
    temperature: 0,
    top_k: 40,
    top_p: 0.95,
    zoom: 1.0,
    max_attempts: 3,
    timeout: 60,
    page_limit: 10
  }
}) => {
  // State for the options
  const [options, setOptions] = useState(defaultOptions);
  const [expanded, setExpanded] = useState(false);
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);
  
  // Multi-prompt state
  const [prompts, setPrompts] = useState([
    {
      id: 1,
      name: 'Table Headers',
      content: TABLE_STRUCTURE_PROMPT,
      isActive: true
    },
    {
      id: 2,
      name: 'Table Names & Totals',
      content: TABLE_NAME_TOTALS_PROMPT,
      isActive: true
    }
  ]);

  const [activePromptIndex, setActivePromptIndex] = useState(0);
  const [nextPromptId, setNextPromptId] = useState(2);
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);
  const [editingPromptName, setEditingPromptName] = useState(null);
  const [tempPromptName, setTempPromptName] = useState('');

  // Use fileId to reset the form when file changes
  useEffect(() => {
    // Optional: Reset the options form when file changes
    // setOptions(defaultOptions);
  }, [fileId]);

  // Update options.prompt when active prompt changes
  useEffect(() => {
    if (prompts[activePromptIndex]) {
      setOptions(prev => ({
        ...prev,
        prompt: prompts[activePromptIndex].content
      }));
    }
  }, [activePromptIndex, prompts]);

  // Available model choices
  const modelChoices = [
    { value: 'mistralVL', label: 'mistral-small3.1:24b' },
    { value: 'gemma3', label: 'gemma3:27b' },
    { value: 'llava7b', label: 'llava:7b' },
  ];

  // Handle option changes
  const handleOptionChange = (field, value) => {
    setOptions(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Handle prompt content changes
  const handlePromptChange = (content) => {
    const updatedPrompts = [...prompts];
    updatedPrompts[activePromptIndex].content = content;
    setPrompts(updatedPrompts);
    handleOptionChange('prompt', content);
  };

  // Add new prompt
  const handleAddPrompt = () => {
    const newPrompt = {
      id: nextPromptId,
      name: `Prompt ${nextPromptId}`,
      content: 'Enter your prompt here...',
      isActive: true
    };
    setPrompts([...prompts, newPrompt]);
    setActivePromptIndex(prompts.length);
    setNextPromptId(nextPromptId + 1);
  };

  // Delete prompt
  const handleDeletePrompt = (index) => {
    if (prompts.length <= 1) return; // Don't delete the last prompt
    
    const updatedPrompts = prompts.filter((_, i) => i !== index);
    setPrompts(updatedPrompts);
    
    // Adjust active index if needed
    if (activePromptIndex >= updatedPrompts.length) {
      setActivePromptIndex(updatedPrompts.length - 1);
    } else if (activePromptIndex > index) {
      setActivePromptIndex(activePromptIndex - 1);
    }
    
    setMenuAnchorEl(null);
  };

  // Toggle prompt active state
  const handleTogglePrompt = (index) => {
    const updatedPrompts = [...prompts];
    updatedPrompts[index].isActive = !updatedPrompts[index].isActive;
    setPrompts(updatedPrompts);
  };

  // Start editing prompt name
  const handleEditPromptName = (index) => {
    setEditingPromptName(index);
    setTempPromptName(prompts[index].name);
    setMenuAnchorEl(null);
  };

  // Save prompt name
  const handleSavePromptName = () => {
    if (editingPromptName !== null) {
      const updatedPrompts = [...prompts];
      updatedPrompts[editingPromptName].name = tempPromptName || `Prompt ${editingPromptName + 1}`;
      setPrompts(updatedPrompts);
      setEditingPromptName(null);
      setTempPromptName('');
    }
  };

  // Cancel editing prompt name
  const handleCancelEditPromptName = () => {
    setEditingPromptName(null);
    setTempPromptName('');
  };

  // Handle running inference with single prompt
  const handleRunInference = () => {
    const currentPrompt = prompts[activePromptIndex];
    if (currentPrompt && currentPrompt.isActive) {
      onRunVisionInference({
        ...options,
        prompt: currentPrompt.content,
        promptName: currentPrompt.name,
        promptId: currentPrompt.id
      });
    }
  };

  // Handle running inference with all active prompts
  const handleRunAllPrompts = () => {
    const activePrompts = prompts.filter(prompt => prompt.isActive);
    if (activePrompts.length > 0) {
      onRunVisionInference({
        ...options,
        prompts: activePrompts.map(prompt => ({
          id: prompt.id,
          name: prompt.name,
          content: prompt.content
        })),
        runMultiple: true
      });
    }
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
      a.download = `vision_inference_${fileId}.json`;
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
  const getCurrentPrompt = () => prompts[activePromptIndex]?.content || '';
  const promptFormat = detectFormat(getCurrentPrompt());

  // Component for rendering the prompt in the appropriate format
  const PromptPreview = () => {
    const currentPromptContent = getCurrentPrompt();
    
    switch (promptFormat) {
      case 'json':
        try {
          const formattedJson = JSON.stringify(JSON.parse(currentPromptContent), null, 2);
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
          return <pre>{currentPromptContent}</pre>;
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
              {currentPromptContent}
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
            <pre style={{ margin: 0 }}>{currentPromptContent}</pre>
          </Box>
        );
    }
  };
  
  // Function to get number of tables safely
  const getTableCount = () => {
    if (!visionResponse || !Array.isArray(visionResponse) || visionResponse.length === 0) {
      return 0;
    }
    return visionResponse[0].number_of_tables || 0;
  };

  // Get count of active prompts
  const activePromptCount = prompts.filter(prompt => prompt.isActive).length;

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
            <Badge badgeContent={prompts.length} color="primary" sx={{ ml: 1 }}>
              <Chip label="prompts" size="small" variant="outlined" />
            </Badge>
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

          {/* Multi-Prompt Section */}
          <Box sx={{ mt: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                Prompts
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
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
                <Button
                  startIcon={<AddIcon />}
                  onClick={handleAddPrompt}
                  size="small"
                  variant="outlined"
                >
                  Add Prompt
                </Button>
              </Box>
            </Box>

            {/* Prompt Tabs */}
            <Paper variant="outlined" sx={{ mb: 2 }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center' }}>
                <Tabs 
                  value={activePromptIndex} 
                  onChange={(_, newValue) => setActivePromptIndex(newValue)}
                  variant="scrollable"
                  scrollButtons="auto"
                  sx={{ flexGrow: 1 }}
                >
                  {prompts.map((prompt, index) => (
                    <Tab
                      key={prompt.id}
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {editingPromptName === index ? (
                            <TextField
                              value={tempPromptName}
                              onChange={(e) => setTempPromptName(e.target.value)}
                              onBlur={handleSavePromptName}
                              onKeyPress={(e) => {
                                if (e.key === 'Enter') handleSavePromptName();
                                if (e.key === 'Escape') handleCancelEditPromptName();
                              }}
                              size="small"
                              autoFocus
                              sx={{ minWidth: '100px' }}
                            />
                          ) : (
                            <>
                              <Box 
                                sx={{ 
                                  width: 8, 
                                  height: 8, 
                                  borderRadius: '50%', 
                                  bgcolor: prompt.isActive ? 'success.main' : 'grey.400' 
                                }} 
                              />
                              {prompt.name}
                            </>
                          )}
                        </Box>
                      }
                      sx={{ 
                        minHeight: 48,
                        opacity: prompt.isActive ? 1 : 0.6
                      }}
                    />
                  ))}
                </Tabs>
                <IconButton
                  onClick={(e) => setMenuAnchorEl(e.currentTarget)}
                  size="small"
                  sx={{ mr: 1 }}
                >
                  <MoreVertIcon />
                </IconButton>
              </Box>

              {/* Prompt Content */}
              <Box sx={{ p: 2 }}>
                {isPreviewMode ? (
                  <PromptPreview />
                ) : (
                  <TextField
                    multiline
                    minRows={4}
                    maxRows={20}
                    value={getCurrentPrompt()}
                    onChange={(e) => handlePromptChange(e.target.value)}
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
            </Paper>

            {/* Menu for prompt actions */}
            <Menu
              anchorEl={menuAnchorEl}
              open={Boolean(menuAnchorEl)}
              onClose={() => setMenuAnchorEl(null)}
            >
              <MenuItem onClick={() => handleTogglePrompt(activePromptIndex)}>
                <ListItemIcon>
                  <Box 
                    sx={{ 
                      width: 16, 
                      height: 16, 
                      borderRadius: '50%', 
                      bgcolor: prompts[activePromptIndex]?.isActive ? 'success.main' : 'grey.400' 
                    }} 
                  />
                </ListItemIcon>
                <ListItemText>
                  {prompts[activePromptIndex]?.isActive ? 'Deactivate' : 'Activate'}
                </ListItemText>
              </MenuItem>
              <MenuItem onClick={() => handleEditPromptName(activePromptIndex)}>
                <ListItemIcon>
                  <EditIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Rename</ListItemText>
              </MenuItem>
              <MenuItem 
                onClick={() => handleDeletePrompt(activePromptIndex)}
                disabled={prompts.length <= 1}
              >
                <ListItemIcon>
                  <DeleteIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Delete</ListItemText>
              </MenuItem>
            </Menu>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        {/* Run Current Prompt */}
        <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
          <span style={{ flex: 1 }}>
            <Button
              variant="contained"
              color="secondary"
              onClick={handleRunInference}
              disabled={visionLoading || !fileId || !hasClassification || !prompts[activePromptIndex]?.isActive}
              startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
              fullWidth
            >
              {visionLoading ? "Running..." : `Run "${prompts[activePromptIndex]?.name || 'Current'}"`}
            </Button>
          </span>
        </Tooltip>

        {/* Run All Active Prompts */}
        <Tooltip title={`Run all ${activePromptCount} active prompts`}>
          <span style={{ flex: 1 }}>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleRunAllPrompts}
              disabled={visionLoading || !fileId || !hasClassification || activePromptCount === 0}
              startIcon={<PlaylistPlayIcon />}
              fullWidth
            >
              Run All ({activePromptCount})
            </Button>
          </span>
        </Tooltip>
      </Box>

      {/* Debug File ID display */}
      {fileId && (
        <Typography variant="caption" sx={{ mt: 1, display: 'block', color: 'text.secondary' }}>
          Current File ID: {fileId}
        </Typography>
      )}

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
                    label={`${getTableCount()} tables detected`} 
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


// import React, { useState, useEffect } from 'react';
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
//   Switch,
//   Paper,
//   Divider,
//   Chip,
//   Card,
//   CardContent,
//   CardHeader,
//   Tabs,
//   Tab,
//   Badge,
//   Menu,
//   ListItemIcon,
//   ListItemText
// } from '@mui/material';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import SettingsIcon from '@mui/icons-material/Settings';
// import CodeIcon from '@mui/icons-material/Code';
// import ContentCopyIcon from '@mui/icons-material/ContentCopy';
// import DownloadIcon from '@mui/icons-material/Download';
// import AddIcon from '@mui/icons-material/Add';
// import DeleteIcon from '@mui/icons-material/Delete';
// import EditIcon from '@mui/icons-material/Edit';
// import PlayArrowIcon from '@mui/icons-material/PlayArrow';
// import PlaylistPlayIcon from '@mui/icons-material/PlaylistPlay';
// import MoreVertIcon from '@mui/icons-material/MoreVert';
// import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// import { atomDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
// import ReactMarkdown from 'react-markdown';
// import remarkGfm from 'remark-gfm';

// // Table Structure Instruction Prompt
// export const TABLE_STRUCTURE_PROMPT = `
// Count how many complete, visually distinct tables appear in the image.
// - Output this count as "number_of_tables".
// - If no tables are identified, output "number_of_tables": 0.

// For each table, determine whether it has hierarchical headers (multi-level columns).
// - A table has hierarchical headers if there is at least one "parent" column spanning multiple sub-columns.
// - Set "headers_hierarchy": true if the table has multi-level headers, otherwise false.

// Return the columns of each table as the lowest-level column names in a flat list.

// Include each table's details under a "tables" key in the output, using the table index (as a string) as the key.

// Example Output:

// {
//   "number_of_tables": 2,
//   "tables": {
//     "0": {
//       "headers_hierarchy": true,
//       "columns": ["Col 1", "Col 2", "Col 3", "Col 4"]
//     },
//     "1": {
//       "headers_hierarchy": false,
//       "columns": ["Col X", "Col Y", "Col Z"]
//     }
//   }
// }
// `;

// // Table Name/Total Row Prompt
// export const TABLE_NAME_TOTALS_PROMPT = `
// Given an image of a PDF page, identify all complete, visually distinct tables.

// For each table, extract the following information:
// 1. Table Name: The title or label of the table if present. If no explicit name is found, use null.
// 2. Columns: A flat list of the lowest-level column names.
// 3. Has Totals Row: Indicate whether the table includes a totals, summary, or grand total row at the bottom. Set "has_totals_row": true if such a row is present, otherwise false.

// Include each table’s details under a "tables" key in the output, using the table index (as a string) as the key.

// Output the number of detected tables as "number_of_tables".

// Example Output:

// {
//   "number_of_tables": 2,
//   "tables": {
//     "0": {
//       "table_name": "Sales Summary",
//       "columns": ["Product", "Quantity", "Unit Price", "Total"],
//       "has_totals_row": true
//     },
//     "1": {
//       "table_name": null,
//       "columns": ["Date", "Description", "Amount"],
//       "has_totals_row": false
//     }
//   }
// }
// `;

// const VisionInferenceOptions = ({
//   fileId,
//   hasClassification,
//   visionLoading,
//   onRunVisionInference,
//   visionResponse = null,
//   defaultOptions = {
//     model_choice: 'mistralVL',
//     temperature: 0,
//     top_k: 40,
//     top_p: 0.95,
//     zoom: 1.0,
//     max_attempts: 3,
//     timeout: 60,
//     page_limit: 10
//   }
// }) => {
//   // State for the options
//   const [options, setOptions] = useState(defaultOptions);
//   const [expanded, setExpanded] = useState(false);
//   const [isPreviewMode, setIsPreviewMode] = useState(false);
//   const [isDarkMode, setIsDarkMode] = useState(true);
//   const [copySuccess, setCopySuccess] = useState(false);
  
//   // Multi-prompt state
//   const [prompts, setPrompts] = useState([
//     {
//       id: 1,
//       name: 'Table Headers',
//       content: TABLE_STRUCTURE_PROMPT,
//       isActive: true
//     },
//     {
//       id: 2,
//       name: 'Table Totals',
//       content: TABLE_NAME_TOTALS_PROMPT,
//       isActive: true
//     }
//   ]);
//   const [activePromptIndex, setActivePromptIndex] = useState(0);
//   const [nextPromptId, setNextPromptId] = useState(2);
//   const [menuAnchorEl, setMenuAnchorEl] = useState(null);
//   const [editingPromptName, setEditingPromptName] = useState(null);
//   const [tempPromptName, setTempPromptName] = useState('');

//   // Use fileId to reset the form when file changes
//   useEffect(() => {
//     // Optional: Reset the options form when file changes
//     // setOptions(defaultOptions);
//   }, [fileId]);

//   // Update options.prompt when active prompt changes
//   useEffect(() => {
//     if (prompts[activePromptIndex]) {
//       setOptions(prev => ({
//         ...prev,
//         prompt: prompts[activePromptIndex].content
//       }));
//     }
//   }, [activePromptIndex, prompts]);

//   // Available model choices
//   const modelChoices = [
//     { value: 'mistralVL', label: 'mistral-small3.1:24b' },
//     { value: 'gemma3', label: 'gemma3:27b' },
//     { value: 'llava7b', label: 'llava:7b' },
//   ];

//   // Handle option changes
//   const handleOptionChange = (field, value) => {
//     setOptions(prev => ({
//       ...prev,
//       [field]: value
//     }));
//   };

//   // Handle prompt content changes
//   const handlePromptChange = (content) => {
//     const updatedPrompts = [...prompts];
//     updatedPrompts[activePromptIndex].content = content;
//     setPrompts(updatedPrompts);
//     handleOptionChange('prompt', content);
//   };

//   // Add new prompt
//   const handleAddPrompt = () => {
//     const newPrompt = {
//       id: nextPromptId,
//       name: `Prompt ${nextPromptId}`,
//       content: 'Enter your prompt here...',
//       isActive: true
//     };
//     setPrompts([...prompts, newPrompt]);
//     setActivePromptIndex(prompts.length);
//     setNextPromptId(nextPromptId + 1);
//   };

//   // Delete prompt
//   const handleDeletePrompt = (index) => {
//     if (prompts.length <= 1) return; // Don't delete the last prompt
    
//     const updatedPrompts = prompts.filter((_, i) => i !== index);
//     setPrompts(updatedPrompts);
    
//     // Adjust active index if needed
//     if (activePromptIndex >= updatedPrompts.length) {
//       setActivePromptIndex(updatedPrompts.length - 1);
//     } else if (activePromptIndex > index) {
//       setActivePromptIndex(activePromptIndex - 1);
//     }
    
//     setMenuAnchorEl(null);
//   };

//   // Toggle prompt active state
//   const handleTogglePrompt = (index) => {
//     const updatedPrompts = [...prompts];
//     updatedPrompts[index].isActive = !updatedPrompts[index].isActive;
//     setPrompts(updatedPrompts);
//   };

//   // Start editing prompt name
//   const handleEditPromptName = (index) => {
//     setEditingPromptName(index);
//     setTempPromptName(prompts[index].name);
//     setMenuAnchorEl(null);
//   };

//   // Save prompt name
//   const handleSavePromptName = () => {
//     if (editingPromptName !== null) {
//       const updatedPrompts = [...prompts];
//       updatedPrompts[editingPromptName].name = tempPromptName || `Prompt ${editingPromptName + 1}`;
//       setPrompts(updatedPrompts);
//       setEditingPromptName(null);
//       setTempPromptName('');
//     }
//   };

//   // Cancel editing prompt name
//   const handleCancelEditPromptName = () => {
//     setEditingPromptName(null);
//     setTempPromptName('');
//   };

//   // Handle running inference with single prompt
//   const handleRunInference = () => {
//     const currentPrompt = prompts[activePromptIndex];
//     if (currentPrompt && currentPrompt.isActive) {
//       onRunVisionInference({
//         ...options,
//         prompt: currentPrompt.content,
//         promptName: currentPrompt.name
//       });
//     }
//   };

//   // Handle running inference with all active prompts
//   const handleRunAllPrompts = () => {
//     const activePrompts = prompts.filter(prompt => prompt.isActive);
//     if (activePrompts.length > 0) {
//       onRunVisionInference({
//         ...options,
//         prompts: activePrompts.map(prompt => ({
//           name: prompt.name,
//           content: prompt.content
//         })),
//         runMultiple: true
//       });
//     }
//   };

//   // Copy response to clipboard
//   const handleCopyResponse = () => {
//     if (visionResponse) {
//       navigator.clipboard.writeText(JSON.stringify(visionResponse, null, 2));
//       setCopySuccess(true);
//       setTimeout(() => setCopySuccess(false), 2000);
//     }
//   };

//   // Download response as JSON file
//   const handleDownloadResponse = () => {
//     if (visionResponse) {
//       const blob = new Blob([JSON.stringify(visionResponse, null, 2)], { type: 'application/json' });
//       const url = URL.createObjectURL(blob);
//       const a = document.createElement('a');
//       a.href = url;
//       a.download = `vision_inference_${fileId}.json`;
//       document.body.appendChild(a);
//       a.click();
//       document.body.removeChild(a);
//       URL.revokeObjectURL(url);
//     }
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
//   const getCurrentPrompt = () => prompts[activePromptIndex]?.content || '';
//   const promptFormat = detectFormat(getCurrentPrompt());

//   // Component for rendering the prompt in the appropriate format
//   const PromptPreview = () => {
//     const currentPromptContent = getCurrentPrompt();
    
//     switch (promptFormat) {
//       case 'json':
//         try {
//           const formattedJson = JSON.stringify(JSON.parse(currentPromptContent), null, 2);
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
//           return <pre>{currentPromptContent}</pre>;
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
//               {currentPromptContent}
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
//             <pre style={{ margin: 0 }}>{currentPromptContent}</pre>
//           </Box>
//         );
//     }
//   };
  
//   // Function to get number of tables safely
//   const getTableCount = () => {
//     if (!visionResponse || !Array.isArray(visionResponse) || visionResponse.length === 0) {
//       return 0;
//     }
//     return visionResponse[0].number_of_tables || 0;
//   };

//   // Get count of active prompts
//   const activePromptCount = prompts.filter(prompt => prompt.isActive).length;

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
//             <Badge badgeContent={prompts.length} color="primary" sx={{ ml: 1 }}>
//               <Chip label="prompts" size="small" variant="outlined" />
//             </Badge>
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
//               label="Page Limit"
//               type="number"
//               value={options.page_limit}
//               onChange={(e) => handleOptionChange('page_limit', parseInt(e.target.value))}
//               fullWidth
//               size="small"
//               inputProps={{ min: 1, max: 100 }}
//               helperText="Maximum number of pages to process"
//             />
//           </Box>

//           {/* Multi-Prompt Section */}
//           <Box sx={{ mt: 3 }}>
//             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
//               <Typography variant="h6" sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
//                 Prompts
//               </Typography>
//               <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
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
//                 <Button
//                   startIcon={<AddIcon />}
//                   onClick={handleAddPrompt}
//                   size="small"
//                   variant="outlined"
//                 >
//                   Add Prompt
//                 </Button>
//               </Box>
//             </Box>

//             {/* Prompt Tabs */}
//             <Paper variant="outlined" sx={{ mb: 2 }}>
//               <Box sx={{ borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center' }}>
//                 <Tabs 
//                   value={activePromptIndex} 
//                   onChange={(_, newValue) => setActivePromptIndex(newValue)}
//                   variant="scrollable"
//                   scrollButtons="auto"
//                   sx={{ flexGrow: 1 }}
//                 >
//                   {prompts.map((prompt, index) => (
//                     <Tab
//                       key={prompt.id}
//                       label={
//                         <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
//                           {editingPromptName === index ? (
//                             <TextField
//                               value={tempPromptName}
//                               onChange={(e) => setTempPromptName(e.target.value)}
//                               onBlur={handleSavePromptName}
//                               onKeyPress={(e) => {
//                                 if (e.key === 'Enter') handleSavePromptName();
//                                 if (e.key === 'Escape') handleCancelEditPromptName();
//                               }}
//                               size="small"
//                               autoFocus
//                               sx={{ minWidth: '100px' }}
//                             />
//                           ) : (
//                             <>
//                               <Box 
//                                 sx={{ 
//                                   width: 8, 
//                                   height: 8, 
//                                   borderRadius: '50%', 
//                                   bgcolor: prompt.isActive ? 'success.main' : 'grey.400' 
//                                 }} 
//                               />
//                               {prompt.name}
//                             </>
//                           )}
//                         </Box>
//                       }
//                       sx={{ 
//                         minHeight: 48,
//                         opacity: prompt.isActive ? 1 : 0.6
//                       }}
//                     />
//                   ))}
//                 </Tabs>
//                 <IconButton
//                   onClick={(e) => setMenuAnchorEl(e.currentTarget)}
//                   size="small"
//                   sx={{ mr: 1 }}
//                 >
//                   <MoreVertIcon />
//                 </IconButton>
//               </Box>

//               {/* Prompt Content */}
//               <Box sx={{ p: 2 }}>
//                 {isPreviewMode ? (
//                   <PromptPreview />
//                 ) : (
//                   <TextField
//                     multiline
//                     minRows={4}
//                     maxRows={20}
//                     value={getCurrentPrompt()}
//                     onChange={(e) => handlePromptChange(e.target.value)}
//                     fullWidth
//                     variant="outlined"
//                     size="small"
//                     InputProps={{
//                       sx: {
//                         height: 'auto',
//                         fontFamily: 'monospace'
//                       }
//                     }}
//                     placeholder="Enter your prompt here..."
//                   />
//                 )}
//               </Box>
//             </Paper>

//             {/* Menu for prompt actions */}
//             <Menu
//               anchorEl={menuAnchorEl}
//               open={Boolean(menuAnchorEl)}
//               onClose={() => setMenuAnchorEl(null)}
//             >
//               <MenuItem onClick={() => handleTogglePrompt(activePromptIndex)}>
//                 <ListItemIcon>
//                   <Box 
//                     sx={{ 
//                       width: 16, 
//                       height: 16, 
//                       borderRadius: '50%', 
//                       bgcolor: prompts[activePromptIndex]?.isActive ? 'success.main' : 'grey.400' 
//                     }} 
//                   />
//                 </ListItemIcon>
//                 <ListItemText>
//                   {prompts[activePromptIndex]?.isActive ? 'Deactivate' : 'Activate'}
//                 </ListItemText>
//               </MenuItem>
//               <MenuItem onClick={() => handleEditPromptName(activePromptIndex)}>
//                 <ListItemIcon>
//                   <EditIcon fontSize="small" />
//                 </ListItemIcon>
//                 <ListItemText>Rename</ListItemText>
//               </MenuItem>
//               <MenuItem 
//                 onClick={() => handleDeletePrompt(activePromptIndex)}
//                 disabled={prompts.length <= 1}
//               >
//                 <ListItemIcon>
//                   <DeleteIcon fontSize="small" />
//                 </ListItemIcon>
//                 <ListItemText>Delete</ListItemText>
//               </MenuItem>
//             </Menu>
//           </Box>
//         </AccordionDetails>
//       </Accordion>

//       {/* Action Buttons */}
//       <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
//         {/* Run Current Prompt */}
//         <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
//           <span style={{ flex: 1 }}>
//             <Button
//               variant="contained"
//               color="secondary"
//               onClick={handleRunInference}
//               disabled={visionLoading || !fileId || !hasClassification || !prompts[activePromptIndex]?.isActive}
//               startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
//               fullWidth
//             >
//               {visionLoading ? "Running..." : `Run "${prompts[activePromptIndex]?.name || 'Current'}"`}
//             </Button>
//           </span>
//         </Tooltip>

//         {/* Run All Active Prompts */}
//         <Tooltip title={`Run all ${activePromptCount} active prompts`}>
//           <span style={{ flex: 1 }}>
//             <Button
//               variant="outlined"
//               color="secondary"
//               onClick={handleRunAllPrompts}
//               disabled={visionLoading || !fileId || !hasClassification || activePromptCount === 0}
//               startIcon={<PlaylistPlayIcon />}
//               fullWidth
//             >
//               Run All ({activePromptCount})
//             </Button>
//           </span>
//         </Tooltip>
//       </Box>

//       {/* Debug File ID display */}
//       {fileId && (
//         <Typography variant="caption" sx={{ mt: 1, display: 'block', color: 'text.secondary' }}>
//           Current File ID: {fileId}
//         </Typography>
//       )}

//       {/* Response Display Section */}
//       {visionResponse && (
//         <Card 
//           variant="outlined" 
//           sx={{ 
//             mt: 3, 
//             overflow: 'visible',
//             borderRadius: 1,
//             boxShadow: '0 2px 10px rgba(0,0,0,0.08)'
//           }}
//         >
//           <CardHeader
//             title={
//               <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
//                 <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                   <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
//                     Vision Inference Result
//                   </Typography>
//                   <Chip 
//                     label={`${getTableCount()} tables detected`} 
//                     size="small" 
//                     color="primary" 
//                     sx={{ ml: 2 }}
//                   />
//                 </Box>
//                 <Box>
//                   <Tooltip title="Toggle Color Theme">
//                     <Switch
//                       checked={isDarkMode}
//                       onChange={(e) => setIsDarkMode(e.target.checked)}
//                       size="small"
//                     />
//                   </Tooltip>
//                   <Tooltip title={copySuccess ? "Copied!" : "Copy JSON"}>
//                     <IconButton onClick={handleCopyResponse} size="small" sx={{ ml: 1 }}>
//                       <ContentCopyIcon fontSize="small" />
//                     </IconButton>
//                   </Tooltip>
//                   <Tooltip title="Download JSON">
//                     <IconButton onClick={handleDownloadResponse} size="small">
//                       <DownloadIcon fontSize="small" />
//                     </IconButton>
//                   </Tooltip>
//                 </Box>
//               </Box>
//             }
//             sx={{ 
//               pb: 0,
//               borderBottom: '1px solid rgba(0,0,0,0.08)'
//             }}
//           />
//           <CardContent sx={{ p: 0 }}>
//             <SyntaxHighlighter 
//               language="json" 
//               style={isDarkMode ? atomDark : oneLight}
//               customStyle={{ 
//                 margin: 0, 
//                 borderRadius: '0 0 4px 4px', 
//                 maxHeight: '500px',
//                 fontSize: '0.9rem'
//               }}
//             >
//               {JSON.stringify(visionResponse, null, 2)}
//             </SyntaxHighlighter>
//           </CardContent>
//         </Card>
//       )}
//     </Box>
//   );
// };

// export default VisionInferenceOptions;

// // // 3. UPDATED VisionInferenceOptions.jsx
// // // The component that displays and tracks per-file responses
// // import React, { useState, useEffect } from 'react';
// // import {
// //   Box,
// //   Button,
// //   CircularProgress,
// //   Tooltip,
// //   Typography,
// //   TextField,
// //   MenuItem,
// //   Accordion,
// //   AccordionSummary,
// //   AccordionDetails,
// //   Slider,
// //   IconButton,
// //   FormControlLabel,
// //   Switch,
// //   Paper,
// //   Divider,
// //   Chip,
// //   Card,
// //   CardContent,
// //   CardHeader
// // } from '@mui/material';
// // import VisibilityIcon from '@mui/icons-material/Visibility';
// // import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// // import SettingsIcon from '@mui/icons-material/Settings';
// // import CodeIcon from '@mui/icons-material/Code';
// // import ContentCopyIcon from '@mui/icons-material/ContentCopy';
// // import DownloadIcon from '@mui/icons-material/Download';
// // import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// // import { atomDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
// // import ReactMarkdown from 'react-markdown';
// // import remarkGfm from 'remark-gfm';


// // const VisionInferenceOptions = ({
// //   fileId,
// //   hasClassification,
// //   visionLoading,
// //   onRunVisionInference,
// //   visionResponse = null, // NEW: Accept response as prop
// //   defaultOptions = {
// //     model_choice: 'gemma3',
// //     temperature: 0,
// //     top_k: 40,
// //     top_p: 0.95,
// //     zoom: 1.0,
// //     max_attempts: 3,
// //     timeout: 60,
// //     prompt: 'Analyze this document page and extract all relevant information.'
// //   }
// // }) => {
// //   // State for the options
// //   const [options, setOptions] = useState(defaultOptions);
// //   const [expanded, setExpanded] = useState(false);
// //   const [isPreviewMode, setIsPreviewMode] = useState(false);
// //   const [isDarkMode, setIsDarkMode] = useState(true);
// //   const [copySuccess, setCopySuccess] = useState(false);
  
// //   // Use fileId to reset the form when file changes
// //   useEffect(() => {
// //     // Optional: Reset the options form when file changes
// //     // setOptions(defaultOptions);
// //   }, [fileId]);

// //   // Available model choices
// //   const modelChoices = [
// //     { value: 'mistralVL', label: 'mistral-small3.1:24b' },
// //     { value: 'gemma3', label: 'gemma3:27b' },
// //     { value: 'llava7b', label: 'llava:7b' },
// //   ];

// //   // Handle option changes
// //   const handleOptionChange = (field, value) => {
// //     setOptions(prev => ({
// //       ...prev,
// //       [field]: value
// //     }));
// //   };

// //   // Handle running inference with the current options
// //   const handleRunInference = () => {
// //     // Call parent component's handler with options
// //     onRunVisionInference(options);
// //   };

// //   // Copy response to clipboard
// //   const handleCopyResponse = () => {
// //     if (visionResponse) {
// //       navigator.clipboard.writeText(JSON.stringify(visionResponse, null, 2));
// //       setCopySuccess(true);
// //       setTimeout(() => setCopySuccess(false), 2000);
// //     }
// //   };

// //   // Download response as JSON file
// //   const handleDownloadResponse = () => {
// //     if (visionResponse) {
// //       const blob = new Blob([JSON.stringify(visionResponse, null, 2)], { type: 'application/json' });
// //       const url = URL.createObjectURL(blob);
// //       const a = document.createElement('a');
// //       a.href = url;
// //       a.download = `vision_inference_${fileId}.json`;
// //       document.body.appendChild(a);
// //       a.click();
// //       document.body.removeChild(a);
// //       URL.revokeObjectURL(url);
// //     }
// //   };

// //   // Detect if the prompt might be JSON or Markdown
// //   const detectFormat = (text) => {
// //     if (text.trim().startsWith('{') && text.trim().endsWith('}')) {
// //       try {
// //         JSON.parse(text);
// //         return 'json';
// //       } catch (e) {
// //         // Not valid JSON
// //       }
// //     }
    
// //     // Check for markdown indicators (headings, lists, etc.)
// //     if (text.match(/^#+\s|^\*\s|-\s\[|^>\s|\*\*.+\*\*|__.+__/m)) {
// //       return 'markdown';
// //     }
    
// //     return 'plaintext';
// //   };

// //   // Get the prompt format
// //   const promptFormat = detectFormat(options.prompt);

// //   // Component for rendering the prompt in the appropriate format
// //   const PromptPreview = () => {
// //     switch (promptFormat) {
// //       case 'json':
// //         try {
// //           const formattedJson = JSON.stringify(JSON.parse(options.prompt), null, 2);
// //           return (
// //             <SyntaxHighlighter 
// //               language="json" 
// //               style={atomDark}
// //               customStyle={{ borderRadius: '4px', maxHeight: '400px' }}
// //             >
// //               {formattedJson}
// //             </SyntaxHighlighter>
// //           );
// //         } catch (e) {
// //           return <pre>{options.prompt}</pre>;
// //         }
// //       case 'markdown':
// //         return (
// //           <Box sx={{ 
// //             p: 2, 
// //             border: '1px solid rgba(0, 0, 0, 0.12)', 
// //             borderRadius: '4px',
// //             backgroundColor: '#f5f5f5',
// //             maxHeight: '400px',
// //             overflow: 'auto'
// //           }}>
// //             <ReactMarkdown 
// //               remarkPlugins={[remarkGfm]}
// //               components={{
// //                 // Custom component for code blocks to add syntax highlighting
// //                 code: ({node, inline, className, children, ...props}) => {
// //                   const match = /language-(\w+)/.exec(className || '');
// //                   return !inline && match ? (
// //                     <SyntaxHighlighter
// //                       language={match[1]}
// //                       style={atomDark}
// //                       PreTag="div"
// //                       {...props}
// //                     >
// //                       {String(children).replace(/\n$/, '')}
// //                     </SyntaxHighlighter>
// //                   ) : (
// //                     <code className={className} {...props}>
// //                       {children}
// //                     </code>
// //                   );
// //                 }
// //               }}
// //             >
// //               {options.prompt}
// //             </ReactMarkdown>
// //           </Box>
// //         );
// //       default:
// //         return (
// //           <Box sx={{ 
// //             p: 2, 
// //             border: '1px solid rgba(0, 0, 0, 0.12)', 
// //             borderRadius: '4px',
// //             backgroundColor: '#f5f5f5',
// //             maxHeight: '400px',
// //             overflow: 'auto',
// //             whiteSpace: 'pre-wrap'
// //           }}>
// //             <pre style={{ margin: 0 }}>{options.prompt}</pre>
// //           </Box>
// //         );
// //     }
// //   };
  
// //   // Function to get number of tables safely
// //   const getTableCount = () => {
// //     if (!visionResponse || !Array.isArray(visionResponse) || visionResponse.length === 0) {
// //       return 0;
// //     }
// //     return visionResponse[0].number_of_tables || 0;
// //   };

// //   return (
// //     <Box sx={{ width: '100%' }}>
// //       <Accordion 
// //         expanded={expanded} 
// //         onChange={() => setExpanded(!expanded)}
// //         sx={{ mb: 2, boxShadow: 'none', border: '1px solid rgba(0, 0, 0, 0.12)' }}
// //       >
// //         <AccordionSummary
// //           expandIcon={<ExpandMoreIcon />}
// //           aria-controls="vision-options-content"
// //           id="vision-options-header"
// //           sx={{ 
// //             '&.Mui-expanded': { minHeight: 48 },
// //             '& .MuiAccordionSummary-content.Mui-expanded': { margin: '12px 0' }
// //           }}
// //         >
// //           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
// //             <SettingsIcon fontSize="small" />
// //             <Typography>Vision Inference Options</Typography>
// //           </Box>
// //         </AccordionSummary>
// //         <AccordionDetails>
// //           <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
// //             {/* Model Selection */}
// //             <TextField
// //               select
// //               label="Model"
// //               value={options.model_choice}
// //               onChange={(e) => handleOptionChange('model_choice', e.target.value)}
// //               fullWidth
// //               size="small"
// //               helperText="Select vision model"
// //             >
// //               {modelChoices.map((option) => (
// //                 <MenuItem key={option.value} value={option.value}>
// //                   {option.label}
// //                 </MenuItem>
// //               ))}
// //             </TextField>

// //             {/* Temperature */}
// //             <Box>
// //               <Typography variant="body2" gutterBottom>Temperature: {options.temperature}</Typography>
// //               <Slider
// //                 value={options.temperature}
// //                 min={0}
// //                 max={1}
// //                 step={0.1}
// //                 onChange={(_, value) => handleOptionChange('temperature', value)}
// //                 aria-labelledby="temperature-slider"
// //               />
// //             </Box>

// //             {/* Top K */}
// //             <TextField
// //               label="Top K"
// //               type="number"
// //               value={options.top_k}
// //               onChange={(e) => handleOptionChange('top_k', parseInt(e.target.value))}
// //               fullWidth
// //               size="small"
// //               inputProps={{ min: 1, max: 100 }}
// //             />

// //             {/* Top P */}
// //             <TextField
// //               label="Top P"
// //               type="number"
// //               value={options.top_p}
// //               onChange={(e) => handleOptionChange('top_p', parseFloat(e.target.value))}
// //               fullWidth
// //               size="small"
// //               inputProps={{ min: 0, max: 1, step: 0.01 }}
// //             />

// //             {/* Zoom */}
// //             <TextField
// //               label="Zoom"
// //               type="number"
// //               value={options.zoom}
// //               onChange={(e) => handleOptionChange('zoom', parseFloat(e.target.value))}
// //               fullWidth
// //               size="small"
// //               inputProps={{ min: 0.1, max: 5, step: 0.1 }}
// //             />

// //             {/* Max Attempts */}
// //             <TextField
// //               label="Max Attempts"
// //               type="number"
// //               value={options.max_attempts}
// //               onChange={(e) => handleOptionChange('max_attempts', parseInt(e.target.value))}
// //               fullWidth
// //               size="small"
// //               inputProps={{ min: 1, max: 10 }}
// //             />

// //             {/* Timeout */}
// //             <TextField
// //               label="Timeout (seconds)"
// //               type="number"
// //               value={options.timeout}
// //               onChange={(e) => handleOptionChange('timeout', parseInt(e.target.value))}
// //               fullWidth
// //               size="small"
// //               inputProps={{ min: 10, max: 300 }}
// //             />

// //             <TextField
// //                 label="Page Limit"
// //                 type="number"
// //                 value={options.page_limit}
// //                 onChange={(e) => handleOptionChange('page_limit', parseInt(e.target.value))}
// //                 fullWidth
// //                 size="small"
// //                 inputProps={{ min: 1, max: 100 }}
// //                 helperText="Maximum number of pages to process"
// //             />
// //           </Box>

// //           {/* Prompt Section with Editor/Preview Toggle */}
// //           <Box sx={{ mt: 2, mb: 1 }}>
// //             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
// //               <Typography variant="body1">Prompt</Typography>
// //               <Box sx={{ display: 'flex', alignItems: 'center' }}>
// //                 <FormControlLabel
// //                   control={
// //                     <Switch
// //                       checked={isPreviewMode}
// //                       onChange={(e) => setIsPreviewMode(e.target.checked)}
// //                       size="small"
// //                     />
// //                   }
// //                   label={
// //                     <Box sx={{ display: 'flex', alignItems: 'center' }}>
// //                       <CodeIcon fontSize="small" sx={{ mr: 0.5 }} />
// //                       <Typography variant="body2">Preview</Typography>
// //                     </Box>
// //                   }
// //                 />
// //               </Box>
// //             </Box>

// //             {isPreviewMode ? (
// //               <PromptPreview />
// //             ) : (
// //               <TextField
// //                 multiline
// //                 minRows={3}
// //                 maxRows={20}
// //                 value={options.prompt}
// //                 onChange={(e) => handleOptionChange('prompt', e.target.value)}
// //                 fullWidth
// //                 variant="outlined"
// //                 size="small"
// //                 InputProps={{
// //                   sx: {
// //                     height: 'auto',
// //                     fontFamily: 'monospace'
// //                   }
// //                 }}
// //                 placeholder="Enter your prompt here..."
// //               />
// //             )}
// //           </Box>
// //         </AccordionDetails>
// //       </Accordion>

// //       {/* Run Vision Button */}
// //       <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
// //         <span>
// //           <Button
// //             variant="contained"
// //             color="secondary"
// //             onClick={handleRunInference}
// //             disabled={visionLoading || !fileId || !hasClassification}
// //             startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
// //             fullWidth
// //           >
// //             {visionLoading ? "Running Vision Inference..." : "Run Vision Inference"}
// //           </Button>
// //         </span>
// //       </Tooltip>


// //     {/* Debug File ID display */}
// //     {fileId && (
// //         <Typography variant="caption" sx={{ mt: 1, display: 'block', color: 'text.secondary' }}>
// //           Current File ID: {fileId}
// //         </Typography>
// //       )}

// //       {/* Response Display Section */}
// //       {visionResponse && (
// //         <Card 
// //           variant="outlined" 
// //           sx={{ 
// //             mt: 3, 
// //             overflow: 'visible',
// //             borderRadius: 1,
// //             boxShadow: '0 2px 10px rgba(0,0,0,0.08)'
// //           }}
// //         >
// //           <CardHeader
// //             title={
// //               <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
// //                 <Box sx={{ display: 'flex', alignItems: 'center' }}>
// //                   <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 600 }}>
// //                     Vision Inference Result
// //                   </Typography>
// //                   <Chip 
// //                     label={`${getTableCount()} tables detected`} 
// //                     size="small" 
// //                     color="primary" 
// //                     sx={{ ml: 2 }}
// //                   />
// //                 </Box>
// //                 <Box>
// //                   <Tooltip title="Toggle Color Theme">
// //                     <Switch
// //                       checked={isDarkMode}
// //                       onChange={(e) => setIsDarkMode(e.target.checked)}
// //                       size="small"
// //                     />
// //                   </Tooltip>
// //                   <Tooltip title={copySuccess ? "Copied!" : "Copy JSON"}>
// //                     <IconButton onClick={handleCopyResponse} size="small" sx={{ ml: 1 }}>
// //                       <ContentCopyIcon fontSize="small" />
// //                     </IconButton>
// //                   </Tooltip>
// //                   <Tooltip title="Download JSON">
// //                     <IconButton onClick={handleDownloadResponse} size="small">
// //                       <DownloadIcon fontSize="small" />
// //                     </IconButton>
// //                   </Tooltip>
// //                 </Box>
// //               </Box>
// //             }
// //             sx={{ 
// //               pb: 0,
// //               borderBottom: '1px solid rgba(0,0,0,0.08)'
// //             }}
// //           />
// //           <CardContent sx={{ p: 0 }}>
// //             <SyntaxHighlighter 
// //               language="json" 
// //               style={isDarkMode ? atomDark : oneLight}
// //               customStyle={{ 
// //                 margin: 0, 
// //                 borderRadius: '0 0 4px 4px', 
// //                 maxHeight: '500px',
// //                 fontSize: '0.9rem'
// //               }}
// //             >
// //               {JSON.stringify(visionResponse, null, 2)}
// //             </SyntaxHighlighter>
// //           </CardContent>
// //         </Card>
// //       )}
// //     </Box>
// //   );
// // };

// // export default VisionInferenceOptions;