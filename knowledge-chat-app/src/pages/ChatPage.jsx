import React, { useState, useEffect, useRef, useMemo, Component } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  VStack,
  HStack,
  Text,
  Input,
  Button,
  Avatar,
  Divider,
  Card,
  CardBody,
  CardHeader,
  IconButton,
  useColorModeValue,
  Heading,
  Textarea,
  InputGroup,
  InputRightElement,
  Badge,
  Code,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Tag,
  TagLabel,
  useDisclosure,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Select,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Progress,
  UnorderedList,
  OrderedList,
  ListItem,
  SimpleGrid,
  useToast,
  Icon,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react';
import { 
  IoSend, 
  IoAdd, 
  IoMenu, 
  IoInformationCircle, 
  IoDocumentText, 
  IoChevronDown,
  IoServer,
  IoAnalytics,
  IoCodeSlash,
  IoGrid,
  IoLayers,
  IoExpand,
  IoContract,
  IoCheckmark,
  IoClose,
  IoRefresh,
  IoArrowForward,
  IoCopy, 
  IoCheckmarkDone,
  IoDownload,
  IoCode,
  IoChevronUp,
  IoInformation,
  IoGitBranch,
  IoArrowRedo,
  IoConstruct,
  IoCheckmarkCircle,
  IoGitCompare,
  IoList,
  IoGitCompareOutline,
  IoCodeSlashOutline
} from 'react-icons/io5';
import { LineageGraph } from '../components/LineageGraph';
import { Prism } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { vscDarkPlus, oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { v4 as uuidv4 } from 'uuid';

// Error boundary component to catch rendering errors
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Component Error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert status="error" variant="solid" flexDirection="column" alignItems="center" my={4} p={4} borderRadius="md">
          <AlertIcon boxSize="40px" mr={0} />
          <AlertTitle mt={4} mb={1} fontSize="lg">Component Error</AlertTitle>
          <AlertDescription maxWidth="100%">
            <Text>There was an error rendering this component:</Text>
            <Code colorScheme="red" d="block" whiteSpace="pre-wrap" overflowX="auto" p={2} my={2}>
              {this.state.error && this.state.error.toString()}
            </Code>
            {this.state.errorInfo && (
              <Code colorScheme="gray" d="block" whiteSpace="pre-wrap" overflowX="auto" fontSize="xs" p={2} maxH="200px" overflow="auto">
                {this.state.errorInfo.componentStack}
              </Code>
            )}
          </AlertDescription>
        </Alert>
      );
    }

    return this.props.children;
  }
}

// Update the FormattedMessage component to better handle data architect responses
const FormattedMessage = ({ content }) => {
  // Handle case where content is not a string
  if (typeof content !== 'string') {
    try {
      content = JSON.stringify(content, null, 2);
    } catch (e) {
      content = "Error displaying content";
    }
  }
  
  // Check if content is empty
  if (!content || content.trim() === '') {
    return <Text color="gray.500">No content available</Text>;
  }
  
  // Check if content contains code blocks
  const hasCodeBlocks = content.includes('```');
  
  // If it has code blocks, use the MarkdownContent component for proper code rendering
  if (hasCodeBlocks) {
    return <MarkdownContent content={content} />;
  }
  
  // Check if content contains markdown sections (## or **)
  const hasMarkdown = content.includes('##') || content.includes('**');
  
  if (hasMarkdown) {
    // Split by markdown headers (##)
    const sections = content.split(/##\s+/);
    
    return (
      <VStack align="start" spacing={4} width="100%">
        {sections.map((section, idx) => {
          if (idx === 0 && !section.trim()) return null;
          
          if (idx === 0) {
            // This is the intro text before any headers
            return (
              <Text 
                key={idx}
                fontSize="16px"
                fontFamily="'Merriweather', Georgia, serif"
                lineHeight="1.7"
                color="gray.800"
                whiteSpace="pre-wrap"
                width="100%"
              >
                {section}
              </Text>
            );
          }
          
          // For sections with headers
          const sectionParts = section.split(/\n/);
          const sectionTitle = sectionParts[0];
          const sectionContent = sectionParts.slice(1).join('\n');
          
          return (
            <Box key={idx} width="100%" mt={2}>
              <Heading 
                size="md" 
                color="purple.700"
                fontWeight="600"
                fontFamily="'Playfair Display', Georgia, serif"
                pb={2}
                borderBottom="2px solid"
                borderColor="purple.200"
                width="fit-content"
                fontSize="18px"
                letterSpacing="0.02em"
                mb={3}
              >
                {sectionTitle}
              </Heading>
              <Text 
                fontSize="16px"
                fontFamily="'Merriweather', Georgia, serif"
                lineHeight="1.7"
                color="gray.800"
                whiteSpace="pre-wrap"
                pl={2}
                borderLeft="3px solid"
                borderColor="purple.100"
                width="100%"
              >
                {sectionContent}
              </Text>
            </Box>
          );
        })}
      </VStack>
    );
  } else {
    // Simple text display for non-markdown content
    return (
      <Text 
        fontSize="16px"
        fontFamily="'Merriweather', Georgia, serif"
        lineHeight="1.7"
        color="gray.800"
        whiteSpace="pre-wrap"
        width="100%"
      >
        {content}
      </Text>
    );
  }
};

// Enhanced MarkdownContent component
const MarkdownContent = ({ content }) => {
  // Process content to extract headers and sections for better formatting
  const processContent = (text) => {
    if (!text) return [];
    
    // Split by markdown headers
    const sections = [];
    const headerRegex = /^(#{1,6})\s+(.+)$/gm;
    
    // Also detect section headers like "Schema Information" followed by a newline
    const sectionHeaderRegex = /^([A-Z][A-Za-z\s]+)(\n|$)/gm;
    
    let lastIndex = 0;
    let headerMatches = [];
    
    // Find all standard markdown headers first
    let headerMatch;
    const headerRegexClone = new RegExp(headerRegex);
    while ((headerMatch = headerRegexClone.exec(text)) !== null) {
      headerMatches.push({
        index: headerMatch.index,
        length: headerMatch[0].length,
        level: headerMatch[1].length,
        text: headerMatch[2],
        isMarkdown: true
      });
    }
    
    // Find section headers (capitalized words followed by newline)
    let sectionMatch;
    const sectionRegexClone = new RegExp(sectionHeaderRegex);
    while ((sectionMatch = sectionRegexClone.exec(text)) !== null) {
      // Skip if this is inside a code block
      const textBefore = text.substring(0, sectionMatch.index);
      const codeBlocksStart = (textBefore.match(/```/g) || []).length;
      if (codeBlocksStart % 2 !== 0) continue; // Inside a code block
      
      // Skip if too close to previous header (might be a false positive)
      const tooCloseToLastHeader = headerMatches.some(h => 
        Math.abs(h.index - sectionMatch.index) < 20
      );
      if (tooCloseToLastHeader) continue;
      
      headerMatches.push({
        index: sectionMatch.index,
        length: sectionMatch[0].length,
        level: 2, // Treat as h2
        text: sectionMatch[1].trim(),
        isMarkdown: false
      });
    }
    
    // Sort all headers by their position in the text
    headerMatches.sort((a, b) => a.index - b.index);
    
    // Process headers and their content
    if (headerMatches.length > 0) {
      for (let i = 0; i < headerMatches.length; i++) {
        const currentHeader = headerMatches[i];
        
        // Add text before this header if exists
        if (currentHeader.index > lastIndex) {
          const contentBeforeHeader = text.substring(lastIndex, currentHeader.index).trim();
          if (contentBeforeHeader) {
            sections.push({
              type: 'text',
              content: contentBeforeHeader
            });
          }
        }
        
        // Find where this section ends (next header or end of text)
        const nextHeader = headerMatches[i + 1];
        const endIndex = nextHeader ? nextHeader.index : text.length;
        
        // Content after header until next header or end
        let sectionContent = '';
        if (currentHeader.index + currentHeader.length < endIndex) {
          sectionContent = text.substring(currentHeader.index + currentHeader.length, endIndex).trim();
        }
        
        // Add this header and its content
        sections.push({
          type: 'header',
          level: currentHeader.level,
          text: currentHeader.text,
          content: sectionContent,
          isMarkdown: currentHeader.isMarkdown
        });
        
        lastIndex = nextHeader ? nextHeader.index : endIndex;
      }
    } else {
      // No headers, just add all content as text
      sections.push({
        type: 'text',
        content: text.trim()
      });
    }
    
    return sections;
  };

  // Better detection and rendering of different content types
  const renderContent = (content) => {
    if (!content) return null;
    
    // First check for code blocks - improved pattern to handle code blocks with no language specified
    const codeBlockRegex = /```([\w]*)(?:\n|\r\n|\r)([\s\S]*?)```/g;
    let match;
    let lastIndex = 0;
    const parts = [];
    
    // Extract code blocks
    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const textPart = content.substring(lastIndex, match.index);
        if (textPart.trim()) {
          parts.push({
            type: 'text',
            content: textPart
          });
        }
      }
      
      // Add code block - handle language properly
      const language = match[1]?.trim() || 'text';
      const code = match[2];
      parts.push({
        type: 'code',
        language,
        content: code
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < content.length) {
      const textPart = content.substring(lastIndex);
      if (textPart.trim()) {
        parts.push({
          type: 'text',
          content: textPart
        });
      }
    }
    
    // If we have parts, render them separately
    if (parts.length > 0) {
      return (
        <>
          {parts.map((part, index) => {
            if (part.type === 'code') {
              return (
                <CodeBlock 
                  key={index} 
                  code={part.content} 
                  language={part.language} 
                />
              );
            } else {
              // Process the text part
              const processedText = processPanels(part.content);
              
              // Check if it's a table
              if (processedText.includes('|') && processedText.includes('\n|')) {
                return <Box key={index}>{renderTable(processedText)}</Box>;
              }
              
              // Otherwise render as mixed content
              return <Box key={index}>{renderMixedContent(processedText)}</Box>;
            }
          })}
        </>
      );
    }
    
    // If no code blocks found, process normally
    content = processPanels(content);
    
    // Process tables
    if (content.includes('|') && content.includes('\n|')) {
      return renderTable(content);
    }
    
    // Handle lists and paragraphs
    return renderMixedContent(content);
  };
  
  // Process panel blocks (info, note, warning, tip)
  const processPanels = (text) => {
    // Look for panel patterns like [INFO], [NOTE], [WARNING], [TIP]
    const panelPatterns = [
      { tag: '[INFO]', className: 'info-panel' },
      { tag: '[NOTE]', className: 'note-panel' },
      { tag: '[WARNING]', className: 'warning-panel' },
      { tag: '[TIP]', className: 'tip-panel' }
    ];
    
    let processedText = text;
    
    // Replace panel tags with HTML classes
    panelPatterns.forEach(({ tag, className }) => {
      const tagRegex = new RegExp(`\\${tag}\\s*(.+?)(?=\\[(?:INFO|NOTE|WARNING|TIP)\\]|$)`, 'gs');
      processedText = processedText.replace(tagRegex, `<div class="${className}">$1</div>`);
    });
    
    return processedText;
  };
  
  // Render tables from markdown pipe syntax
  const renderTable = (text) => {
    // Extract table lines
    const tableLines = text.split('\n').filter(line => line.trim().startsWith('|'));
    
    if (tableLines.length < 2) {
      return renderMixedContent(text);
    }
    
    // Extract header row
    const headerLine = tableLines[0];
    const headers = headerLine.split('|')
      .map(cell => cell.trim())
      .filter(cell => cell !== '');
    
    // Skip separator row
    const dataRows = tableLines.slice(2);
    
    return (
      <Box className="key-value-table" overflow="auto">
        <Table size="sm" variant="simple">
          <Thead>
            <Tr>
              {headers.map((header, i) => (
                <Th key={i}>{header}</Th>
              ))}
            </Tr>
          </Thead>
          <Tbody>
            {dataRows.map((row, rowIdx) => {
              const cells = row.split('|')
                .map(cell => cell.trim())
                .filter(cell => cell !== '');
              
              return (
                <Tr key={rowIdx}>
                  {cells.map((cell, cellIdx) => (
                    <Td key={cellIdx}>{cell}</Td>
                  ))}
                </Tr>
              );
            })}
          </Tbody>
        </Table>
      </Box>
    );
  };
  
  // Render mixed content with lists, paragraphs, etc.
  const renderMixedContent = (text) => {
    if (!text) return null;
    
    // Process schema-specific formatting (bolded column names like **column_name**:)
    const processedText = text.replace(/\*\*([^*]+)\*\*\s*:/g, '<strong class="schema-field">$1</strong>:');
    
    // Handle HTML panel divs
    if (processedText.includes('<div class="')) {
      const parts = processedText.split(/(<div class=".*?">.*?<\/div>)/gs);
      
      return (
        <>
          {parts.map((part, index) => {
            if (part.startsWith('<div class="') && part.endsWith('</div>')) {
              // Extract class and content
              const classMatch = part.match(/<div class="(.*?)">(.*?)<\/div>/s);
              if (classMatch) {
                const className = classMatch[1];
                const panelContent = classMatch[2];
                
                return (
                  <Box key={index} className={className}>
                    {renderListItems(panelContent)}
                  </Box>
                );
              }
            }
            
            return part.trim() ? renderListItems(part) : null;
          })}
        </>
      );
    }
    
    // Check if this is schema information with field definitions
    const isSchemaInfo = /\*\*.*?\*\*\s*:/.test(processedText);
    
    if (isSchemaInfo) {
      return (
        <Box className="schema-section">
          {renderListItems(processedText)}
        </Box>
      );
    }
    
    // Regular content
    return renderListItems(processedText);
  };
  
  // Improved list item detection and rendering
  const renderListItems = (text) => {
    if (!text) return null;
    
    // First, convert bold text formatting (**text**)
    let formattedText = text;
    
    // Replace **text** with proper bold formatting
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Split text into lines
    const lines = formattedText.split('\n');
    
    // Group lines into list items or paragraphs
    const elements = [];
    let currentList = null;
    let currentParagraph = '';
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();
      
      // Check if line is a bullet list item
      const bulletMatch = trimmedLine.match(/^[-*]\s+(.+)$/);
      if (bulletMatch) {
        // Finish any current paragraph
        if (currentParagraph) {
          elements.push({
            type: 'paragraph',
            content: currentParagraph.trim()
          });
          currentParagraph = '';
        }
        
        // Start a new list if needed
        if (!currentList || currentList.type !== 'bullet') {
          if (currentList) {
            elements.push(currentList);
          }
          currentList = {
            type: 'bullet',
            items: []
          };
        }
        
        currentList.items.push(bulletMatch[1]);
        continue;
      }
      
      // Check if line is a numbered list item
      const numberedMatch = trimmedLine.match(/^\d+\.\s+(.+)$/);
      if (numberedMatch) {
        // Finish any current paragraph
        if (currentParagraph) {
          elements.push({
            type: 'paragraph',
            content: currentParagraph.trim()
          });
          currentParagraph = '';
        }
        
        // Start a new list if needed
        if (!currentList || currentList.type !== 'numbered') {
          if (currentList) {
            elements.push(currentList);
          }
          currentList = {
            type: 'numbered',
            items: []
          };
        }
        
        currentList.items.push(numberedMatch[1]);
        continue;
      }
      
      // Regular line - finish any current list
      if (currentList) {
        elements.push(currentList);
        currentList = null;
      }
      
      // Add to current paragraph or start a new one
      if (trimmedLine === '' && currentParagraph) {
        elements.push({
          type: 'paragraph',
          content: currentParagraph.trim()
        });
        currentParagraph = '';
      } else if (trimmedLine !== '') {
        currentParagraph += (currentParagraph ? '\n' : '') + line;
      }
    }
    
    // Add any remaining list or paragraph
    if (currentList) {
      elements.push(currentList);
    }
    
    if (currentParagraph) {
      elements.push({
        type: 'paragraph',
        content: currentParagraph.trim()
      });
    }
    
    // Render all elements
    return (
      <>
        {elements.map((element, index) => {
          if (element.type === 'paragraph') {
            return <Text key={index} dangerouslySetInnerHTML={{ __html: element.content }} />;
          } else if (element.type === 'bullet') {
            return (
              <UnorderedList key={index} pl={4} spacing={1} my={2}>
                {element.items.map((item, itemIndex) => (
                  <ListItem key={itemIndex} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              </UnorderedList>
            );
          } else if (element.type === 'numbered') {
            return (
              <OrderedList key={index} pl={4} spacing={1} my={2}>
                {element.items.map((item, itemIndex) => (
                  <ListItem key={itemIndex} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              </OrderedList>
            );
          }
          return null;
        })}
      </>
    );
  };

  const sections = processContent(content);
  
  return (
    <Box>
      {sections.map((section, idx) => {
        if (section.type === 'header') {
          return (
            <Box key={idx} mt={4} mb={2}>
              <Heading 
                as={`h${section.level}`} 
                size={section.level <= 2 ? "md" : "sm"}
                color="purple.700"
                pb={1}
                mb={2}
                borderBottom={section.level <= 2 ? "1px solid" : "none"}
                borderColor="purple.100"
              >
                {section.text}
              </Heading>
              <Box pl={2}>
                {renderContent(section.content)}
              </Box>
            </Box>
          );
        } else {
          return (
            <Box key={idx} my={2}>
              {renderContent(section.content)}
            </Box>
          );
        }
      })}
    </Box>
  );
};

// Update the CodeBlock component with improved styling
const CodeBlock = ({ code, language }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `snippet.${language || 'txt'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  return (
    <Box
      position="relative"
      my={4}
      borderRadius="md"
      overflow="hidden"
      fontSize="sm"
      boxShadow="lg"
      className="code-block"
      bg="#282c34"
      border="1px solid"
      borderColor="gray.700"
      width="100%"
    >
      <HStack
        bg="#21252b"
        color="gray.100"
        p={2}
        justify="space-between"
        align="center"
        className="code-header"
        borderBottom="1px solid"
        borderColor="gray.700"
      >
        <Badge colorScheme="blue" variant="solid">
          {language || 'code'}
        </Badge>
        <HStack>
          <IconButton
            icon={copied ? <IoCheckmarkDone /> : <IoCopy />}
            size="sm"
            variant="ghost"
            color="gray.300"
            _hover={{ bg: 'gray.700' }}
            colorScheme={copied ? "green" : "gray"}
            onClick={handleCopy}
            aria-label="Copy code"
            title="Copy to clipboard"
          />
          <IconButton
            icon={<IoDownload />}
            size="sm"
            variant="ghost"
            color="gray.300"
            _hover={{ bg: 'gray.700' }}
            onClick={handleDownload}
            aria-label="Download code"
            title="Download code"
          />
        </HStack>
      </HStack>
      <Box position="relative" className="prism-wrapper">
        <Prism
          language={language || 'text'}
          style={atomDark}
          customStyle={{
            margin: 0,
            padding: '16px',
            maxHeight: '500px',
            overflow: 'auto',
            backgroundColor: '#282c34',
            color: '#abb2bf',
            fontSize: '0.9em',
            border: 'none',
            borderRadius: 0,
            width: '100%'
          }}
        >
          {code}
        </Prism>
      </Box>
    </Box>
  );
};

// Enhanced parser that prioritizes actual repository data
const parseLineageData = (text) => {
  if (!text) return null;
  
  // Create empty data structure
  const data = {
    models: [],
    columns: [],
    edges: []
  };
  
  try {
    // Split text into lines and process
    const lines = text.split('\n').filter(line => line.trim());
    
    let currentModelId = null;
    let highlightedModel = null;
    let currentModelPath = null;
    let modelCount = 0;
    let columnCount = 0;
    
    // First pass: identify all model paths and create a unique-path map
    const modelPaths = new Map(); // Map to track unique model paths
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Skip markdown headers or empty lines
      if (line.startsWith('#') || !line) continue;
      
      // Look for model paths with high precision using regex patterns
      // Focus on actual model paths in the format models/dir/file.sql
      const modelPathPatterns = [
        /\b(models\/[a-zA-Z0-9_/]+\.sql)\b/,                      // Standard paths: models/path/model.sql
        /[├└]──\s*(models\/[a-zA-Z0-9_/]+\.sql)/,                 // Tree notation paths
        /source\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)/ // Source references
      ];
      
      // Special case for tpch models which have specific naming patterns
      const tpchPatterns = [
        /\b(stg_tpch_[a-zA-Z0-9_]+)\.sql\b/,            // stg_tpch_orders.sql
        /\b(stg_tpch_[a-zA-Z0-9_]+)\b/                  // stg_tpch_orders
      ];
      
      // First try standard model path patterns
      let foundMatch = false;
      for (const pattern of modelPathPatterns) {
        const matches = line.match(pattern);
        if (matches) {
          let path;
          if (pattern.toString().includes('source')) {
            // Handle source references
            path = `source:${matches[1]}.${matches[2]}`;
          } else {
            path = matches[1];
          }
          
          // Only add to our map if not already seen
          if (!modelPaths.has(path)) {
            const name = path.includes('/') ? path.split('/').pop().replace('.sql', '') : path.split('.').pop();
            // Determine model type from path
            let type = 'unknown';
            if (path.includes('source:')) {
              type = 'source';
            } else if (path.includes('/staging/')) {
              type = 'staging';
            } else if (path.includes('/intermediate/')) {
              type = 'intermediate';
            } else if (path.includes('/mart') || path.includes('/core/')) {
              type = 'mart';
            }
            
            // Check if this model is likely the highlighted one
            const isHighlighted = line.includes('highlight') || 
                               line.includes('requested_model') ||
                               line.includes('model of interest');
            
            modelPaths.set(path, {
              id: `model${modelPaths.size + 1}`,
              name,
              path,
              type,
              highlight: isHighlighted
            });
            
            foundMatch = true;
          }
        }
      }
      
      // If no standard match was found, try tpch patterns
      if (!foundMatch) {
        for (const pattern of tpchPatterns) {
          const matches = line.match(pattern);
          if (matches) {
            const modelName = matches[1];
            // We know tpch models are in staging/tpch
            const path = `models/staging/tpch/${modelName}.sql`;
            
            if (!modelPaths.has(path)) {
              modelPaths.set(path, {
                id: `model${modelPaths.size + 1}`,
                name: modelName,
                path,
                type: 'staging',
                highlight: false
              });
            }
          }
        }
      }
    }
    
    // Second pass: convert map to array and process relationships
    data.models = Array.from(modelPaths.values());
    
    // If no highlighted model was found, look for clues to identify main model
    if (!data.models.some(m => m.highlight)) {
      // Look for models that appear most frequently or are centered in diagrams
      const modelMentionCount = new Map();
      
      // Direct clues about which model is the main focus
      const requestPhrases = [
        'requested model', 
        'model of interest', 
        'central model',
        'intermediate/order_items',     // Direct reference to order_items
        'marts/intermediate/order_items.sql' // Full path to order_items
      ];
      
      // First check if we have a model named "order_items" since that's a common focus
      const orderItemsModel = data.models.find(m => 
        m.name === 'order_items' || 
        m.path.includes('/order_items.sql') ||
        m.path.includes('intermediate/order_items')
      );
      
      if (orderItemsModel) {
        orderItemsModel.highlight = true;
        highlightedModel = orderItemsModel.id;
      } else {
        // Count model mentions to find the most referenced one
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          
          // First check for direct indicators in the request phrases
          let foundDirectIndicator = false;
          for (const phrase of requestPhrases) {
            if (line.toLowerCase().includes(phrase.toLowerCase())) {
              // Find a model that might match this phrase
              for (const model of data.models) {
                if (
                  (phrase.includes('order_items') && (model.name === 'order_items' || model.path.includes('order_items'))) ||
                  line.includes(model.path) || 
                  line.includes(model.name)
                ) {
                  model.highlight = true;
                  highlightedModel = model.id;
                  foundDirectIndicator = true;
                  break;
                }
              }
              if (foundDirectIndicator) break;
            }
          }
          
          if (foundDirectIndicator) break;
          
          // If no direct indicator, count mentions
          for (const model of data.models) {
            if (line.includes(model.path) || line.includes(model.name)) {
              modelMentionCount.set(model.id, (modelMentionCount.get(model.id) || 0) + 1);
            }
            
            // Look for direct indicators this is the main model
            if (line.includes(`└── ${model.path}`) || 
                line.includes(`→ ${model.path}`) ||
                (line.includes('Detailed Model') && lines[i+1] && lines[i+1].includes(model.path))) {
              model.highlight = true;
              highlightedModel = model.id;
              break;
            }
          }
        }
        
        // If still no highlighted model, use the most mentioned one
        if (!highlightedModel && modelMentionCount.size > 0) {
          const sortedMentions = [...modelMentionCount.entries()].sort((a, b) => b[1] - a[1]);
          if (sortedMentions.length > 0) {
            const [mostMentionedId] = sortedMentions[0];
            const model = data.models.find(m => m.id === mostMentionedId);
            if (model) {
              model.highlight = true;
              highlightedModel = model.id;
            }
          }
        }
      }
    }
    
    // Third pass: extract relationships between models
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Look for relationships indicated by arrows or other patterns
      if (line.includes('→') || line.includes('->')) {
        // Split by arrow
        const parts = line.split(/→|->/).map(part => part.trim());
        
        for (let j = 0; j < parts.length - 1; j++) {
          const sourcePart = parts[j];
          const targetPart = parts[j + 1];
          
          // Find models that match these parts
          const sourceModel = data.models.find(model => 
            sourcePart.includes(model.path) || sourcePart.includes(model.name));
          
          const targetModel = data.models.find(model => 
            targetPart.includes(model.path) || targetPart.includes(model.name));
          
          if (sourceModel && targetModel) {
            // Add edge if not already present
            if (!data.edges.some(e => e.source === sourceModel.id && e.target === targetModel.id)) {
              data.edges.push({
                source: sourceModel.id,
                target: targetModel.id
              });
            }
          }
        }
      }
      
      // Look for explicit upstream/downstream statements
      if (line.toLowerCase().includes('upstream') && i < lines.length - 1) {
        const upstreamLine = lines[i + 1];
        const mainModel = data.models.find(m => m.highlight);
        
        if (mainModel) {
          // Find any model mentioned in the upstream line
          const upstreamModel = data.models.find(model => 
            upstreamLine.includes(model.path) || upstreamLine.includes(model.name));
          
          if (upstreamModel && upstreamModel.id !== mainModel.id) {
            // Add edge from upstream to main
            if (!data.edges.some(e => e.source === upstreamModel.id && e.target === mainModel.id)) {
              data.edges.push({
                source: upstreamModel.id,
                target: mainModel.id
              });
            }
          }
        }
      }
      
      if (line.toLowerCase().includes('downstream') && i < lines.length - 1) {
        const downstreamLine = lines[i + 1];
        const mainModel = data.models.find(m => m.highlight);
        
        if (mainModel) {
          // Find any model mentioned in the downstream line
          const downstreamModel = data.models.find(model => 
            downstreamLine.includes(model.path) || downstreamLine.includes(model.name));
          
          if (downstreamModel && downstreamModel.id !== mainModel.id) {
            // Add edge from main to downstream
            if (!data.edges.some(e => e.source === mainModel.id && e.target === downstreamModel.id)) {
              data.edges.push({
                source: mainModel.id,
                target: downstreamModel.id
              });
            }
          }
        }
      }
    }
    
    // If we have models but no edges, try to infer them from the model types
    if (data.models.length > 0 && data.edges.length === 0) {
      // Sort models by type (source -> staging -> intermediate -> mart)
      const typeOrder = { 'source': 0, 'staging': 1, 'intermediate': 2, 'mart': 3 };
      const sortedModels = [...data.models].sort((a, b) => 
        (typeOrder[a.type] || 999) - (typeOrder[b.type] || 999));
      
      // Create edges between adjacent types
      for (let i = 0; i < sortedModels.length - 1; i++) {
        const current = sortedModels[i];
        const next = sortedModels[i + 1];
        
        // Only create an edge if the next model is at least one level higher
        if ((typeOrder[next.type] || 0) > (typeOrder[current.type] || 0)) {
          data.edges.push({
            source: current.id,
            target: next.id
          });
        }
      }
    }
    
    // If no model is highlighted, highlight the central model
    if (!data.models.some(m => m.highlight) && data.models.length > 0) {
      if (data.edges.length > 0) {
        // Find the model with the most connections
        const connectionCounts = {};
        data.edges.forEach(edge => {
          connectionCounts[edge.source] = (connectionCounts[edge.source] || 0) + 1;
          connectionCounts[edge.target] = (connectionCounts[edge.target] || 0) + 1;
        });
        
        let maxConnections = 0;
        let centralModelId = null;
        
        Object.entries(connectionCounts).forEach(([modelId, count]) => {
          if (count > maxConnections) {
            maxConnections = count;
            centralModelId = modelId;
          }
        });
        
        if (centralModelId) {
          const centralModel = data.models.find(m => m.id === centralModelId);
          if (centralModel) {
            centralModel.highlight = true;
          }
        }
      } else {
        // Just highlight the first model
        data.models[0].highlight = true;
      }
    }
    
    return data;
  } catch (error) {
    console.error("Error parsing lineage data:", error);
    return null;
  }
};

// Update the renderArchitectResponse function to process content properly
const renderArchitectResponse = (content, sections) => {
  // Process the content normally without adding lineage visualization
  return renderContent(content);
};

// Function to render content with markdown
const renderContent = (content) => {
  // If content contains code blocks, use MarkdownContent
  if (content && content.includes('```')) {
    return <MarkdownContent content={content} />;
  }
  
  // If content has markdown formatting
  if (content && (content.includes('##') || content.includes('**'))) {
    return <FormattedMessage content={content} />;
  }
  
  // Use ReactMarkdown for other content
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <SyntaxHighlighter
              {...props}
              style={vscDarkPlus}
              language={match[1]}
              PreTag="div"
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code {...props} className={className}>
              {children}
            </code>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

// Create a separate message component to handle individual message state
const MessageComponent = React.memo(({ message }) => {
  const [showLineage, setShowLineage] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);
  
  const isUser = message.role === 'user';
  const isArchitectResponse = message.role === 'assistant' && message.type === 'architect';
  
  return (
    <Box
      bg={isUser ? 'blue.50' : isArchitectResponse ? 'purple.50' : 'white'}
      p={4}
      borderRadius="md"
      maxWidth={isUser ? '80%' : '95%'}
      alignSelf={isUser ? 'flex-end' : 'flex-start'}
      boxShadow="sm"
      mb={4}
      border="1px solid"
      borderColor={isUser ? 'blue.100' : isArchitectResponse ? 'purple.100' : 'gray.200'}
      width={isUser ? 'auto' : '95%'}
    >
      <VStack align="stretch" spacing={3}>
        <HStack>
          <Avatar 
            size="sm" 
            bg={isUser ? 'blue.500' : isArchitectResponse ? 'purple.500' : 'green.500'} 
            name={isUser ? 'You' : isArchitectResponse ? 'Data Architect' : 'Assistant'} 
          />
          <Text fontWeight="bold" color={
            isUser ? 'blue.700' : 
            isArchitectResponse ? 'purple.700' : 
            'green.700'
          }>
            {isUser ? 'You' : isArchitectResponse ? 'Data Architect' : 'Assistant'}
          </Text>
          
          {isArchitectResponse && message.details?.processing_time && (
            <Badge colorScheme="purple" ml={2}>
              {message.details.processing_time.toFixed(1)}s
            </Badge>
          )}
          
          {isArchitectResponse && message.details?.question_type && (
            <Badge colorScheme="blue" ml={2}>
              {message.details.question_type}
            </Badge>
          )}
        </HStack>
        
        <Box flex="1" className={`confluence-styled-content ${!isUser ? 'code-styled-content' : ''}`}>
          {isUser ? (
            <Text>{message.content}</Text>
          ) : (
            <FormattedMessage content={message.content} />
          )}
        </Box>
        
        {/* Lineage visualization if available */}
        {message.hasLineage && message.lineageData && (
          <ErrorBoundary>
            <Box 
              mt={4} 
              p={4} 
              borderWidth="1px" 
              borderColor="purple.200" 
              borderRadius="md"
              bg="white"
              width="100%"
            >
              <Text fontWeight="bold" mb={3} fontSize="lg">Data Lineage Visualization</Text>
              <Box height="500px">
                <LineageGraph data={message.lineageData} />
              </Box>
            </Box>
          </ErrorBoundary>
        )}
        
        {/* Collapsible JSON Data - Hidden by default */}
        {message.hasLineage && message.lineageData && (
          <Box mt={3} width="100%">
            <Button 
              size="sm" 
              width="100%" 
              onClick={() => setShowRawJson(!showRawJson)}
              variant="outline"
              leftIcon={showRawJson ? <IoChevronUp /> : <IoChevronDown />}
              justifyContent="space-between"
              colorScheme="gray"
            >
              <Text>Lineage JSON</Text>
            </Button>
            
            {showRawJson && (
              <Box 
                mt={2} 
                p={3} 
                borderWidth="1px" 
                borderColor="gray.200" 
                borderRadius="md"
                bg="gray.50"
                maxHeight="400px"
                overflowY="auto"
                width="100%"
              >
                <Code p={3} width="100%" display="block" whiteSpace="pre" overflowX="auto" fontSize="sm">
                  {JSON.stringify(message.lineageData, null, 2)}
                </Code>
              </Box>
            )}
          </Box>
        )}
      </VStack>
    </Box>
  );
});

// Add display name for better debugging
MessageComponent.displayName = 'ChatMessageComponent';

// Update the renderMessage function to use the MessageComponent
const renderMessage = (message) => {
  if (!message) return null;
  return <MessageComponent message={message} />;
};

// Main ChatPage component
const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const messagesEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const messagesContainerRef = useRef(null);

  // Initialize conversation or load from ID - only runs when conversationId changes
  useEffect(() => {
    // Clear messages when component mounts if there's no conversation ID
    if (!conversationId) {
      setMessages([]);
      setCurrentConversationId(null);
    } else if (conversationId !== currentConversationId) {
      // Only fetch if conversation ID changed
      fetchConversation(conversationId);
    }
  }, [conversationId]);

  // Handle scrolling behavior separately, only when messages change
  // and only if autoScroll is enabled
  useEffect(() => {
    if (autoScroll && messagesEndRef.current && messages.length > 0) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.length, autoScroll]);

  // Detect user scroll to disable auto-scrolling
  useEffect(() => {
    const messagesContainer = messagesContainerRef.current;
    if (!messagesContainer) return;

    const handleScroll = () => {
      // If user scrolls up, disable auto-scroll
      const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
      
      // If we're not near the bottom (within 100px), disable auto-scroll
      if (scrollHeight - (scrollTop + clientHeight) > 100) {
        setAutoScroll(false);
      } else {
        setAutoScroll(true);
      }
    };

    messagesContainer.addEventListener('scroll', handleScroll);
    return () => messagesContainer.removeEventListener('scroll', handleScroll);
  }, [messagesContainerRef.current]);

  // Reset auto-scroll when new message is added
  useEffect(() => {
    // When a new message is added and it's from us or there's a loading state change, enable auto-scroll
    if (messages.length > 0 && (messages[messages.length - 1].role === 'assistant' || loading)) {
      setAutoScroll(true);
    }
  }, [messages.length, loading]);

  // Apply confluence styles
  React.useEffect(() => {
    // Create style element
    const styleEl = document.createElement('style');
    let cssText = '';
    
    // Define styles
    const confluenceStyles = {
      '.confluence-styled-content': {
        fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        lineHeight: '1.6',
        color: '#172B4D',
        fontSize: '14px',
      },
      // Regular inline code styling
      '.confluence-styled-content code': {
        backgroundColor: '#F4F5F7',
        padding: '2px 4px',
        borderRadius: '3px',
        fontSize: '13px',
        fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace',
      },
      // Dark code block styling
      '.code-styled-content pre': {
        backgroundColor: '#282c34',
        color: '#abb2bf',
        borderRadius: '4px',
        padding: '16px',
        overflow: 'auto',
        fontSize: '14px',
        fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace',
        marginBottom: '16px',
        marginTop: '16px',
      },
      '.code-styled-content pre code': {
        backgroundColor: 'transparent',
        padding: 0,
        color: '#abb2bf',
        fontSize: '13px',
        fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace',
        border: 'none',
      },
      '.confluence-styled-content h1, .confluence-styled-content h2, .confluence-styled-content h3, .confluence-styled-content h4': {
        fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        fontWeight: '600',
        lineHeight: '1.3',
        margin: '16px 0 8px 0',
        color: '#172B4D',
      },
      '.confluence-styled-content h1': {
        fontSize: '20px',
        borderBottom: '1px solid #DFE1E6',
        paddingBottom: '8px',
      },
      '.confluence-styled-content h2': {
        fontSize: '18px',
        borderBottom: '1px solid #DFE1E6',
        paddingBottom: '6px',
      },
      '.confluence-styled-content h3': {
        fontSize: '16px',
      },
      '.confluence-styled-content h4': {
        fontSize: '14px',
        fontWeight: '600',
      },
      '.confluence-styled-content p': {
        margin: '8px 0',
        lineHeight: '1.6',
      },
      '.confluence-styled-content ul, .confluence-styled-content ol': {
        paddingLeft: '24px',
        margin: '8px 0',
      },
      '.confluence-styled-content li': {
        margin: '4px 0',
      },
      '.confluence-styled-content blockquote': {
        borderLeft: '3px solid #DFE1E6',
        margin: '16px 0',
        padding: '0 16px',
        color: '#5E6C84',
      },
      '.confluence-styled-content table': {
        borderCollapse: 'collapse',
        width: '100%',
        margin: '16px 0',
      },
      '.confluence-styled-content th, .confluence-styled-content td': {
        border: '1px solid #DFE1E6',
        padding: '8px',
        textAlign: 'left',
      },
      '.confluence-styled-content th': {
        backgroundColor: '#F4F5F7',
        fontWeight: '600',
      },
      '.confluence-styled-content img': {
        maxWidth: '100%',
        height: 'auto',
      },
      '.confluence-styled-content hr': {
        border: '0',
        height: '1px',
        backgroundColor: '#DFE1E6',
        margin: '24px 0',
      },
      '.confluence-styled-content a': {
        color: '#0052CC',
        textDecoration: 'none',
      },
      '.confluence-styled-content a:hover': {
        textDecoration: 'underline',
      },
      '.code-panel': {
        margin: '16px 0',
        borderRadius: '3px',
        overflow: 'hidden',
      },
      '.code-panel-header': {
        backgroundColor: '#F4F5F7',
        padding: '8px 16px',
        fontWeight: '600',
        borderBottom: '1px solid #DFE1E6',
      },
      '.code-panel-body': {
        backgroundColor: '#FFFFFF',
        padding: '16px',
        overflowX: 'auto',
      },
      '.info-panel': {
        backgroundColor: '#DEEBFF',
        borderRadius: '3px',
        padding: '16px',
        margin: '16px 0',
        borderLeft: '3px solid #0747A6',
      },
      '.note-panel': {
        backgroundColor: '#EAE6FF',
        borderRadius: '3px',
        padding: '16px',
        margin: '16px 0',
        borderLeft: '3px solid #5243AA',
      },
      '.warning-panel': {
        backgroundColor: '#FFEBE6',
        borderRadius: '3px',
        padding: '16px',
        margin: '16px 0',
        borderLeft: '3px solid #DE350B',
      },
      '.tip-panel': {
        backgroundColor: '#E3FCEF',
        borderRadius: '3px',
        padding: '16px',
        margin: '16px 0',
        borderLeft: '3px solid #00875A',
      },
      '.code-content': {
        backgroundColor: '#282c34',
        position: 'relative',
        zIndex: '1',
        color: '#abb2bf'
      }
    };
    
    // Convert style object to CSS text
    Object.entries(confluenceStyles).forEach(([selector, styles]) => {
      cssText += `${selector} {\n`;
      Object.entries(styles).forEach(([property, value]) => {
        cssText += `  ${property.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value};\n`;
      });
      cssText += '}\n';
    });
    
    styleEl.textContent = cssText;
    document.head.appendChild(styleEl);
    
    // Cleanup function
    return () => {
      document.head.removeChild(styleEl);
    };
  }, []);

  // Fetch a specific conversation
  const fetchConversation = async (id) => {
    try {
      console.log("Fetching conversation with ID:", id);
      
      // Set loading state
      setLoading(true);
      
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching conversation: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation data received:", data);
      
      // Check for various possible data structures
      if (!data) {
        throw new Error("No data received from API");
      }
      
      // The API may return data in different formats, let's handle both possibilities
      const conversation = data.conversation || data;
      
      if (!conversation) {
        console.error("Invalid conversation data structure:", data);
        throw new Error("Invalid conversation data structure received");
      }
      
      console.log("Parsed conversation:", conversation);
      
      // Create properly formatted messages
      const formattedMessages = [];
      
      // Add user message
      if (conversation.query) {
        formattedMessages.push({
          id: `user-${id}`,
          role: 'user',
          content: conversation.query,
          timestamp: conversation.timestamp
        });
      }
      
      // Add assistant message
      if (conversation.response) {
        // Extract lineage data if present
        const { lineageData, cleanedContent } = extractLineageData(conversation.response);
        console.log("Extracted lineage data:", lineageData ? "Found" : "Not found");
        
        formattedMessages.push({
          id: `assistant-${id}`,
          role: 'assistant',
          type: 'architect',
          content: cleanedContent,
          hasLineage: !!lineageData,
          lineageData: lineageData,
          details: {
            ...(conversation.technical_details || {}),
            conversation_id: id
          },
          timestamp: conversation.timestamp
        });
      }
      
      console.log("Formatted messages:", formattedMessages);
      
      // Update state
      setMessages(formattedMessages);
      setCurrentConversationId(id);
      
    } catch (error) {
      console.error('Error fetching conversation:', error);
      toast({
        title: 'Error Loading Conversation',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      // End loading state
      setLoading(false);
    }
  };

  // Function to detect and extract lineage data from response content
  const extractLineageData = (content) => {
    if (!content) {
      console.warn("No content provided for lineage extraction");
      return { lineageData: null, cleanedContent: content };
    }
    
    console.log("Extracting lineage data from content of length:", content.length);
    
    // Create a copy of the content to remove JSON blocks
    let cleanedContent = content;
    let lineageData = null;
    
    // Look for a JSON block with LINEAGE_VISUALIZATION
    try {
      // First try to find a structured JSON block with the exact pattern from the backend logs
      const lineageVizMatch = content.match(/# LINEAGE_VISUALIZATION\s*```json\s*([\s\S]*?)\s*```/);
      
      if (lineageVizMatch && lineageVizMatch[1]) {
        console.log("Found lineage visualization match with pattern '# LINEAGE_VISUALIZATION'");
        try {
          lineageData = JSON.parse(lineageVizMatch[1]);
          console.log("Successfully extracted structured lineage data");
          
          // Remove the JSON block from the content
          cleanedContent = cleanedContent.replace(/# LINEAGE_VISUALIZATION\s*```json\s*[\s\S]*?\s*```/, '');
        } catch (e) {
          console.error("Error parsing lineage visualization JSON:", e);
          // Try to clean the JSON string before parsing
          try {
            const cleanedJson = lineageVizMatch[1].replace(/\n/g, '').trim();
            lineageData = JSON.parse(cleanedJson);
            console.log("Successfully parsed cleaned JSON");
            
            // Remove the JSON block from the content
            cleanedContent = cleanedContent.replace(/# LINEAGE_VISUALIZATION\s*```json\s*[\s\S]*?\s*```/, '');
          } catch (cleanError) {
            console.error("Still failed to parse cleaned JSON:", cleanError);
          }
        }
      } else {
        // Try alternative patterns
        console.log("Trying alternative lineage patterns");
        
        // Pattern 2: Without the # prefix
        const altLineageMatch = content.match(/LINEAGE_VISUALIZATION\s*```json\s*([\s\S]*?)\s*```/);
        if (altLineageMatch && altLineageMatch[1]) {
          console.log("Found match with pattern 'LINEAGE_VISUALIZATION'");
          try {
            lineageData = JSON.parse(altLineageMatch[1]);
            
            // Remove the JSON block from the content
            cleanedContent = cleanedContent.replace(/LINEAGE_VISUALIZATION\s*```json\s*[\s\S]*?\s*```/, '');
          } catch (e) {
            console.error("Error parsing alternative lineage JSON:", e);
          }
        }
        
        // Pattern 3: Looking for just a JSON block with models and edges
        const jsonBlockMatch = content.match(/```json\s*({[\s\S]*?"models"[\s\S]*?"edges"[\s\S]*?})\s*```/);
        if (jsonBlockMatch && jsonBlockMatch[1]) {
          console.log("Found JSON block with models and edges");
          try {
            lineageData = JSON.parse(jsonBlockMatch[1]);
            
            // Remove the JSON block from the content
            cleanedContent = cleanedContent.replace(/```json\s*({[\s\S]*?"models"[\s\S]*?"edges"[\s\S]*?})\s*```/, '');
          } catch (e) {
            console.error("Error parsing JSON block:", e);
          }
        }
        
        // Pattern 4: Try to extract lineage data using our custom parser
        if (!lineageData) {
          lineageData = parseLineageData(content);
        }
      }
    } catch (err) {
      console.error("Error in lineage extraction:", err);
    }
    
    // Clean up any double newlines created by removing blocks
    cleanedContent = cleanedContent.replace(/\n{3,}/g, '\n\n').trim();
    
    // If no lineage data found, return null
    console.log("Lineage data extraction complete:", lineageData ? "Found" : "Not found");
    return { lineageData, cleanedContent };
  };

  // Send message and get response from data architect
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    // Update the messages state with the user message
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);
    setAutoScroll(true); // Enable auto-scroll when sending a new message
    
    try {
      console.log("Sending request to backend with input:", input);
      
      // Use the Data Architect agent endpoint
      const response = await fetch('http://localhost:8000/architect/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          query: input,
          conversation_id: currentConversationId,
          thread_id: currentConversationId
        })
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Data Architect response received:", data);
      
      // Create a conversation ID if needed
      const responseConversationId = data.conversation_id || currentConversationId || crypto.randomUUID();
      
      // Extract lineage data if present
      const { lineageData, cleanedContent } = extractLineageData(data.response);
      console.log("Lineage data extraction result:", lineageData ? "Found lineage data" : "No lineage data found");
      
      if (lineageData) {
        console.log("Lineage data models:", lineageData.models?.length);
        console.log("Lineage data edges:", lineageData.edges?.length);
      }
      
      // Add the Data Architect's response to the chat
      const assistantMessage = {
        role: 'assistant',
        type: 'architect',
        content: cleanedContent,
        id: Date.now(),
        hasLineage: !!lineageData,
        lineageData: lineageData,
        details: {
          conversation_id: responseConversationId,
          question_type: data.question_type,
          processing_time: data.processing_time,
          github_results: data.github_results?.results || [],
          sql_results: data.sql_results?.results || [],
          doc_results: data.doc_results?.results || [],
          dbt_results: data.dbt_results?.results || [],
          relationship_results: data.relationship_results?.results || []
        }
      };
      
      console.log("Adding assistant message with lineage:", !!lineageData);
      
      // Directly update the state with both messages in one update
      setMessages([...updatedMessages, assistantMessage]);
      
      // Update conversation tracking info
      setCurrentConversationId(responseConversationId);
      
      // Update URL if we have a new conversation ID
      if (responseConversationId && (!conversationId || responseConversationId !== conversationId)) {
        navigate(`/chat/${responseConversationId}`, { replace: true });
      }
      
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: 'Error',
        description: 'Failed to send message. Please try again.',
        status: 'error',
        duration: 3000,
      });
      
      // Add error message to chat
      setMessages([...updatedMessages, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        id: Date.now(),
        isError: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Empty state when no messages
  const renderEmptyState = () => (
    <VStack spacing={6} py={10} textAlign="center">
      <Heading size="lg">Welcome to the Data Architect Chat</Heading>
      <Text>Ask questions about your data models, lineage, and more.</Text>
      <Text fontSize="sm" color="gray.500">
        Try asking: "What models depend on stg_orders?" or "Show me the lineage for fct_orders"
      </Text>
    </VStack>
  );

  return (
    <Container maxW="80%" py={4}>
      <Box 
        h="calc(100vh - 170px)" 
        display="flex" 
        flexDirection="column"
      >
        {/* Auto-scroll button - only show when needed */}
        <Box mb={4}>
          {autoScroll ? (
            <Button 
              size="sm" 
              colorScheme="gray" 
              onClick={() => setAutoScroll(false)}
              leftIcon={<IoContract />}
            >
              Auto-scroll On
            </Button>
          ) : (
            <Button 
              size="sm" 
              colorScheme="blue" 
              onClick={() => setAutoScroll(true)}
              leftIcon={<IoExpand />}
            >
              Auto-scroll Off
            </Button>
          )}
        </Box>

        {/* Messages area */}
        <Box 
          ref={messagesContainerRef}
          flex="1" 
          overflowY="auto" 
          px={4} 
          py={4}
          borderWidth="1px"
          borderRadius="md"
          mb={4}
          bg="gray.50"
          position="relative"
        >
          {!autoScroll && messages.length > 3 && (
            <Button
              position="sticky"
              top="10px"
              right="10px"
              zIndex="10"
              size="sm"
              colorScheme="purple"
              opacity="0.8"
              _hover={{ opacity: 1 }}
              onClick={() => {
                setAutoScroll(true);
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              }}
              leftIcon={<IoChevronUp />}
              float="right"
              mr={2}
            >
              Scroll to Bottom
            </Button>
          )}
          
          {messages.length > 0 ? 
            <VStack spacing={4} align="stretch">
              {messages.map(renderMessage)}
              <div ref={messagesEndRef} />
            </VStack>
            : 
            renderEmptyState()
          }
          
          {loading && (
            <Box p={4} bg="blue.50" borderRadius="md" mt={4}>
              <HStack>
                <Icon as={IoAnalytics} color="blue.500" boxSize={5} mr={2} />
                <Text>Processing your request...</Text>
              </HStack>
              <Progress size="xs" colorScheme="blue" isIndeterminate mt={3} />
            </Box>
          )}
        </Box>
        
        {/* Input area */}
        <Box as="form" onSubmit={handleSubmit}>
          <InputGroup size="lg">
            <Input
              placeholder="Ask about data models, lineage, or SQL..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              borderColor="gray.300"
              _focus={{ borderColor: 'purple.500', boxShadow: '0 0 0 1px purple.500' }}
              isDisabled={loading}
            />
            <InputRightElement width="4.5rem">
              <Button 
                h="1.75rem" 
                size="sm" 
                colorScheme="purple" 
                isLoading={loading}
                type="submit"
                leftIcon={<IoSend />}
                disabled={!input.trim() || loading}
              >
                Send
              </Button>
            </InputRightElement>
          </InputGroup>
        </Box>
      </Box>
    </Container>
  );
};

export default ChatPage; 