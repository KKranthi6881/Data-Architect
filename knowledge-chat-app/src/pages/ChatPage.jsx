import React, { useState, useEffect, useRef, useMemo } from 'react'
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
  Icon
} from '@chakra-ui/react'
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
  IoChevronUp
} from 'react-icons/io5'
import { useParams, useNavigate } from 'react-router-dom'
import { CodeDisplay } from '../components/CodeDisplay'
import { Prism } from 'react-syntax-highlighter'
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import { v4 as uuidv4 } from 'uuid'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism'

// Sample chat history data
const chatHistory = [
  {
    id: 1,
    title: "Customer Data Schema",
    preview: "Tell me about the customer data schema",
    timestamp: "2023-11-15T10:30:00Z",
    active: true
  },
  {
    id: 2,
    title: "SQL Query Optimization",
    preview: "How can I optimize this SQL query?",
    timestamp: "2023-11-14T14:45:00Z",
    active: false
  },
  {
    id: 3,
    title: "ETL Pipeline Analysis",
    preview: "Explain our ETL pipeline",
    timestamp: "2023-11-10T09:15:00Z",
    active: false
  }
];

// Sample conversation data for the active chat
const sampleConversation = [
  {
    id: 1,
    role: 'assistant',
    content: "Hello! I'm your Data Architecture Assistant. How can I help you today? You can ask me about database schemas, data models, ETL processes, or query optimization.",
    timestamp: "2023-11-15T10:30:00Z"
  },
  {
    id: 2,
    role: 'user',
    content: "Tell me about the customer data schema",
    timestamp: "2023-11-15T10:31:00Z"
  },
  {
    id: 3,
    role: 'assistant',
    content: "The customer data schema consists of several related tables that store information about customers and their interactions with our platform.",
    timestamp: "2023-11-15T10:31:30Z",
    sources: [
      { title: 'Data Dictionary', type: 'document' },
      { title: 'Snowflake Schema', type: 'database' }
    ],
    tables: [
      { name: 'customers', rowCount: '1.2M', lastUpdated: 'Today, 09:15 AM' },
      { name: 'customer_addresses', rowCount: '1.5M', lastUpdated: 'Today, 09:15 AM' },
      { name: 'customer_preferences', rowCount: '950K', lastUpdated: 'Yesterday, 11:30 PM' }
    ]
  }
];

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
      >
        {content}
      </Text>
    );
  }
};

const ChatMessage = ({ message }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Format the message content sections
  const formatMessageContent = () => {
    const details = message.details?.question_analysis;
    if (!details) return null;

    return (
      <VStack align="start" spacing={3} width="100%">
        {/* Business Understanding */}
        <Box>
          <Text fontWeight="medium">Business Understanding:</Text>
          <Text>{details.rephrased_question}</Text>
        </Box>

        {/* Key Points */}
        <Box>
          <Text fontWeight="medium">Key Business Points:</Text>
          <UnorderedList>
            {details.key_points?.map((point, idx) => (
              <ListItem key={idx}>{point}</ListItem>
            ))}
          </UnorderedList>
        </Box>
        
        {/* Technical Context */}
        <Box>
          <Text fontWeight="medium">Technical Analysis:</Text>
          <Text>{details.technical_context?.analysis || "No technical analysis available"}</Text>
        </Box>
      </VStack>
    );
  };

  return (
    <VStack align="stretch" spacing={4} w="100%">
      {/* Always show the message content */}
      <Box bg="white" p={4} borderRadius="md" shadow="sm">
        <Text whiteSpace="pre-wrap">{message.content}</Text>
      </Box>

      {/* View Details Accordion */}
      {message.role === 'assistant' && message.details && (
        <Accordion allowToggle width="100%">
          <AccordionItem border="none">
            <AccordionButton 
              px={4} 
              py={2}
              bg="gray.50"
              _hover={{ bg: 'gray.100' }}
              borderRadius="md"
            >
              <Box flex="1" textAlign="left">
                <Text fontWeight="medium" color="blue.600">View Analysis Details</Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              {formatMessageContent()}
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      )}
    </VStack>
  );
};

// Add this component for section display
const ArchitectSection = ({ title, content }) => {
  // Extract code blocks if present
  const codeBlocks = content.match(/```(\w+)?\s*([\s\S]*?)```/g) || [];
  const textContent = content.replace(/```(\w+)?\s*([\s\S]*?)```/g, '').trim();

  return (
    <Box mb={6} borderLeft="4px" borderColor="purple.200" pl={4}>
      <Heading size="md" mb={3} color="purple.700">
        {title}
      </Heading>
      
      {textContent && (
        <VStack align="stretch" spacing={2} mb={codeBlocks.length > 0 ? 4 : 0}>
          {textContent.split('\n').map((line, i) => {
            if (line.trim().startsWith('-')) {
              return (
                <HStack key={i} align="start" spacing={2}>
                  <Box color="purple.500">•</Box>
                  <Text>{line.replace('-', '').trim()}</Text>
                </HStack>
              );
            }
            if (line.trim().startsWith('1.') || line.trim().startsWith('2.') || line.trim().startsWith('3.')) {
              return (
                <Text key={i} pl={4} fontWeight="medium">
                  {line.trim()}
                </Text>
              );
            }
            return <Text key={i}>{line}</Text>;
          })}
        </VStack>
      )}

      {codeBlocks.map((block, index) => {
        const [, lang, code] = block.match(/```(\w+)?\s*([\s\S]*?)```/) || [];
        return (
          <CodeDisplay
            key={index}
            code={code.trim()}
            language={lang?.toLowerCase() || 'sql'}
          />
        );
      })}
    </Box>
  );
};

// Update the renderArchitectResponse function to include LineageGraph
const renderArchitectResponse = (content, sections) => {
  // Check for lineage visualization content in sections
  let lineageData = null;
  let lineageSection = null;
  
  // First look for specific lineage section
  ['lineage', 'data_lineage', 'model_lineage'].forEach(sectionName => {
    if (sections[sectionName] && !lineageData) {
      lineageSection = sections[sectionName];
      lineageData = parseLineageData(sections[sectionName]);
    }
  });
  
  // If no specific lineage section was found, try to find lineage data in any section
  if (!lineageData) {
    for (const [name, content] of Object.entries(sections)) {
      // Skip sections that are unlikely to contain lineage info
      if (['thinking', 'summary', 'introduction', 'conclusion'].includes(name.toLowerCase())) continue;
      
      // Check if this section has model references and direction indicators
      if (
        (content.includes('models/') && 
         (content.includes('→') || content.includes('Upstream') || content.includes('Downstream')))
      ) {
        lineageSection = content;
        lineageData = parseLineageData(content);
        break;
      }
    }
  }
  
  // If still no lineage data, check the entire content
  if (!lineageData && 
      (content.includes('models/') && 
       (content.includes('→') || content.includes('Upstream') || content.includes('Downstream')))) {
    lineageData = parseLineageData(content);
    lineageSection = content;
  }
  
  return (
    <Box>
      {lineageData && (
        <VStack w="100%" spacing={4} align="stretch">
          <Box 
            borderWidth="1px" 
            borderStyle="dashed" 
            borderColor="purple.300" 
            borderRadius="md" 
            p={2} 
            bg="purple.50"
            my={4}
          >
            <Text color="purple.700" fontWeight="medium" mb={2}>
              Data Lineage Visualization
            </Text>
            <LineageGraph data={lineageData} />
          </Box>
          
          {lineageSection && (
            <Box 
              bg="gray.50" 
              p={3} 
              borderRadius="md" 
              fontSize="sm" 
              fontFamily="mono"
              overflowX="auto"
            >
              <Text fontWeight="medium" mb={2} color="gray.600">
                Original Lineage Description
              </Text>
              <MarkdownContent content={lineageSection} />
            </Box>
          )}
        </VStack>
      )}
      
      {renderContent(content)}
    </Box>
  );
};

// Update the formatMessageContent function to handle <think> sections
const formatMessageContent = (content) => {
  if (!content) return '';
  
  // First, remove any <think>...</think> or <thinking>...</thinking> sections
  let cleanedContent = content;
  
  // Handle variations of thinking tags with regex
  cleanedContent = cleanedContent.replace(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi, '');
  
  // Check if content contains code blocks with triple backticks
  if (cleanedContent.includes('```')) {
    const parts = [];
    let currentIndex = 0;
    let codeBlockStart = cleanedContent.indexOf('```', currentIndex);
    
    // Process each code block
    while (codeBlockStart !== -1) {
      // Add text before code block
      if (codeBlockStart > currentIndex) {
        parts.push({
          type: 'text',
          content: cleanedContent.substring(currentIndex, codeBlockStart)
        });
      }
      
      // Find the end of the code block
      const codeBlockEnd = cleanedContent.indexOf('```', codeBlockStart + 3);
      if (codeBlockEnd === -1) {
        // No closing backticks, treat rest as text
        parts.push({
          type: 'text',
          content: cleanedContent.substring(codeBlockStart)
        });
        break;
      }
      
      // Extract language and code
      const codeWithLang = cleanedContent.substring(codeBlockStart + 3, codeBlockEnd);
      const firstLineBreak = codeWithLang.indexOf('\n');
      const language = firstLineBreak > 0 ? codeWithLang.substring(0, firstLineBreak).trim() : '';
      const code = firstLineBreak > 0 ? codeWithLang.substring(firstLineBreak + 1) : codeWithLang;
      
      parts.push({
        type: 'code',
        language: language || 'sql', // Default to SQL if no language specified
        content: code
      });
      
      currentIndex = codeBlockEnd + 3;
      codeBlockStart = cleanedContent.indexOf('```', currentIndex);
    }
    
    // Add remaining text after last code block
    if (currentIndex < cleanedContent.length) {
      parts.push({
        type: 'text',
        content: cleanedContent.substring(currentIndex)
      });
    }
    
    return parts;
  }
  
  // If no code blocks, return as single text part
  return [{ type: 'text', content: cleanedContent }];
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
    
    // First check for code blocks
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
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
      
      // Add code block
      const language = match[1] || 'text';
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

// Update the renderMessageContent function to handle lineage visualization
const renderMessageContent = (message) => {
  let content = message.content;
  
  // Remove any "thinking" sections from the content
  content = content.replace(/\n+<thinking>[\s\S]*?<\/thinking>\n+/g, '\n\n');
  
  // For architect responses, check for lineage visualization data
  if (message.role === 'assistant' && message.metadata?.agentType === 'data_architect') {
    // Check if content contains lineage data
    const hasLineageData = content.includes('models/') && 
      (content.includes('→') || content.includes('Upstream:') || 
       content.includes('Downstream:') || content.includes('lineage'));
    
    if (hasLineageData) {
      // Extract sections from content
      const sections = {};
      
      // Look for markdown headers to extract sections
      const headerMatches = [...content.matchAll(/#{1,4}\s+([^\n]+)/g)];
      
      if (headerMatches.length > 0) {
        // Process sections based on headers
        headerMatches.forEach((match, index) => {
          const headerText = match[1].trim();
          const headerPos = match.index;
          const nextHeaderPos = index < headerMatches.length - 1 
            ? headerMatches[index + 1].index 
            : content.length;
          
          // Extract section content
          const sectionContent = content.substring(headerPos, nextHeaderPos)
            .replace(/#{1,4}\s+([^\n]+)/, '') // Remove the header
            .trim();
          
          const sectionKey = headerText.toLowerCase().replace(/\s+/g, '_');
          sections[sectionKey] = sectionContent;
        });
      } else {
        // If no headers found, look for lineage patterns in paragraphs
        const paragraphs = content.split(/\n{2,}/);
        
        for (let i = 0; i < paragraphs.length; i++) {
          const paragraph = paragraphs[i].trim();
          
          if (paragraph.toLowerCase().includes('lineage') || 
              (paragraph.includes('models/') && 
              (paragraph.includes('→') || paragraph.includes('Upstream') || 
               paragraph.includes('Downstream')))) {
            sections['model_lineage'] = paragraph;
            break;
          }
        }
      }
      
      return renderArchitectResponse(content, sections);
    }
  }
  
  // Handle regular markdown content
  if (message.metadata?.format === 'markdown' || message.content.includes('```')) {
    return <MarkdownContent content={content} />;
  }
  
  // Handle code blocks
  const codeBlockRegex = /```(\w+)?\n([\s\S]+?)```/g;
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      parts.push({
        type: 'text',
        content: content.substring(lastIndex, match.index)
      });
    }
    
    // Add code block
    parts.push({
      type: 'code',
      language: match[1] || 'plaintext',
      content: match[2].trim()
    });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < content.length) {
    parts.push({
      type: 'text',
      content: content.substring(lastIndex)
    });
  }
  
  // If parts were split, render them separately
  if (parts.length > 1) {
    return (
      <>
        {parts.map((part, index) => (
          part.type === 'text' ? 
            <MarkdownContent key={`part-${index}`} content={part.content} /> :
            <CodeBlock 
              key={`code-${index}`} 
              code={part.content} 
              language={part.language}
            />
        ))}
      </>
    );
  }
  
  // Fall back to Markdown if only one part or no code blocks
  return <MarkdownContent content={content} />;
};

// Add this component to better display code results

const CodeResultDisplay = ({ result }) => {
  if (!result) return null;
  
  const { file, repository, content } = result;
  
  return (
    <Box 
      borderWidth="1px" 
      borderRadius="md" 
      p={3} 
      mb={3}
      bg="gray.50"
    >
      <HStack mb={2}>
        <Text fontWeight="bold">{file?.path || 'Unknown file'}</Text>
        {repository && (
          <Badge colorScheme="purple">
            {repository.name}
          </Badge>
        )}
      </HStack>
      
      <Code p={2} borderRadius="md" fontSize="sm" overflowX="auto" whiteSpace="pre">
        {content || 'No content available'}
      </Code>
      
      {result.dbt_info && (
        <Box mt={2} p={2} bg="purple.50" borderRadius="md">
          <Text fontWeight="bold" mb={1}>DBT Model Info</Text>
          {result.dbt_info.model_name && (
            <Text fontSize="sm">Model: {result.dbt_info.model_name}</Text>
          )}
          {result.dbt_info.materialization && (
            <Text fontSize="sm">Materialization: {result.dbt_info.materialization}</Text>
          )}
          {result.dbt_info.references && result.dbt_info.references.length > 0 && (
            <Text fontSize="sm">References: {result.dbt_info.references.join(', ')}</Text>
          )}
        </Box>
      )}
    </Box>
  );
};

// Update the renderMessage function to use the new rendering for technical content
const renderMessage = (message) => {
  if (!message) return null;
  
  const isUser = message.role === 'user';
  const isArchitectResponse = message.role === 'assistant' && message.type === 'architect';
  
  return (
    <Box
      key={message.id}
      bg={isUser ? 'blue.50' : isArchitectResponse ? 'purple.50' : 'white'}
      p={4}
      borderRadius="md"
      maxWidth={isUser ? '70%' : '90%'}
      alignSelf={isUser ? 'flex-end' : 'flex-start'}
      boxShadow="sm"
      mb={4}
      border="1px solid"
      borderColor={isUser ? 'blue.100' : isArchitectResponse ? 'purple.100' : 'gray.200'}
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
        
        <Box flex="1" className="confluence-styled-content">
          {renderMessageContent(message)}
        </Box>
        
        {/* Show GitHub results */}
        {isArchitectResponse && message.details?.github_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      GitHub Code Results ({message.details.github_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.github_results.map((result, idx) => (
                      <CodeResultDisplay key={idx} result={result} />
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
        
        {/* Show SQL results */}
        {isArchitectResponse && message.details?.sql_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      SQL Schema Results ({message.details.sql_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.sql_results.map((result, idx) => (
                      <Box 
                        key={idx}
                        p={3}
                        bg="gray.50"
                        borderRadius="md"
                        borderLeft="3px solid"
                        borderColor="blue.300"
                      >
                        <Text fontWeight="medium">{result.metadata?.source || 'SQL Schema'}</Text>
                        <Code p={2} mt={2} fontSize="sm" overflowX="auto" whiteSpace="pre">
                          {result.content || 'No SQL content available'}
                        </Code>
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
        
        {/* Show DBT results */}
        {isArchitectResponse && message.details?.dbt_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      DBT Model Results ({message.details.dbt_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.dbt_results.map((result, idx) => (
                      <Box key={idx} p={4} borderWidth="1px" borderRadius="md">
                        <Text fontWeight="medium">{result.model_name}</Text>
                        <Text fontSize="sm" color="gray.500">{result.description}</Text>
                        {result.dependencies && (
                          <Box mt={2}>
                            <Text fontSize="sm" fontWeight="medium">Dependencies:</Text>
                            <UnorderedList fontSize="sm" ml={4}>
                              {result.dependencies.map((dep, depIdx) => (
                                <ListItem key={depIdx}>{dep}</ListItem>
                              ))}
                            </UnorderedList>
                          </Box>
                        )}
                        {result.sql && (
                          <Box mt={2}>
                            <Text fontSize="sm" fontWeight="medium">SQL:</Text>
                            <Code display="block" p={2} borderRadius="md" fontSize="sm">
                              {result.sql}
                            </Code>
                          </Box>
                        )}
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
      </VStack>
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
            maxHeight: '400px',
            overflow: 'auto',
            backgroundColor: '#282c34',
            color: '#abb2bf',
            fontSize: '0.9em',
            border: 'none',
            borderRadius: 0
          }}
        >
          {code}
        </Prism>
      </Box>
    </Box>
  );
};

// Add additional styles to improve readability as a technical document
const confluenceStyles = {
  '.confluence-styled-content': {
    fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    lineHeight: '1.6',
    color: '#172B4D',
    fontSize: '14px',
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
  '.confluence-styled-content code': {
    backgroundColor: '#F4F5F7',
    padding: '2px 4px',
    borderRadius: '3px',
    fontSize: '13px',
    fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace',
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
  '.key-value-table': {
    display: 'table',
    width: '100%',
    borderCollapse: 'collapse',
    marginBottom: '16px',
  },
  '.key-value-row': {
    display: 'table-row',
  },
  '.key-cell': {
    display: 'table-cell',
    padding: '8px',
    backgroundColor: '#F4F5F7',
    fontWeight: '600',
    border: '1px solid #DFE1E6',
    width: '30%',
  },
  '.value-cell': {
    display: 'table-cell',
    padding: '8px',
    border: '1px solid #DFE1E6',
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
  },
  '.code-content pre': {
    margin: 0,
    backgroundColor: '#282c34',
    color: '#abb2bf'
  },
  '.code-content code': {
    backgroundColor: 'transparent',
    color: '#abb2bf'
  },
  '.code-content .prism-code': {
    background: '#282c34 !important',
    color: '#abb2bf !important'
  },
  '.confluence-styled-content strong': {
    fontWeight: '600',
    color: '#172B4D',
  },
  // Additional schema-specific styling
  '.schema-section strong': {
    fontWeight: '600',
    color: '#0747A6',
    display: 'inline-block',
    marginTop: '8px',
  },
  // Make code blocks in Prism stand out more
  '.prism-code': {
    backgroundColor: '#282c34 !important',
    color: '#abb2bf !important',
    border: 'none !important',
    borderRadius: '4px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
  },
  // Style inline backtick code
  'code:not(.prism-code)': {
    backgroundColor: '#282c34',
    color: '#abb2bf',
    borderRadius: '3px',
    padding: '2px 4px',
    fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace',
    fontSize: '90%'
  },
  // Improve table styling
  'table': {
    width: '100%',
    borderCollapse: 'collapse',
    marginBottom: '16px',
    fontSize: '14px',
  },
  'th': {
    backgroundColor: '#F4F5F7',
    fontWeight: '600',
    padding: '8px 12px',
    borderBottom: '2px solid #DFE1E6',
    textAlign: 'left',
  },
  'td': {
    padding: '8px 12px',
    borderBottom: '1px solid #DFE1E6',
    verticalAlign: 'top',
  },
  // Fix for Prism container 
  '.react-syntax-highlighter-line-number': {
    backgroundColor: '#282c34 !important',
  },
  '.schema-field': {
    fontWeight: '600',
    color: '#0747A6',
    marginTop: '8px',
    display: 'inline-block',
  },
  '.schema-section': {
    padding: '0',
    marginBottom: '8px',
  },
  '.confluence-styled-content pre, .confluence-styled-content code': {
    fontFamily: '"SFMono-Medium", "SF Mono", "Segoe UI Mono", "Roboto Mono", "Ubuntu Mono", Menlo, Consolas, Courier, monospace',
  },
  '.confluence-styled-content pre': {
    backgroundColor: '#1e1e1e',  // Dark background for code blocks
    color: '#f8f8f2',            // Light text for code blocks
    padding: '12px 16px',
    borderRadius: '4px',
    overflow: 'auto',
    marginBottom: '16px',
    fontSize: '13px',
    lineHeight: '1.4',
    border: '1px solid #333'
  },
  '.confluence-styled-content code': {
    backgroundColor: '#f4f5f7',
    color: '#172b4d',
    padding: '2px 4px',
    borderRadius: '3px',
    fontSize: '0.9em',
  },
  '.code-block': {
    fontFamily: '"SFMono-Medium", "SF Mono", "Segoe UI Mono", "Roboto Mono", "Ubuntu Mono", Menlo, Consolas, Courier, monospace',
    backgroundColor: '#282c34',
    color: '#abb2bf',
    borderRadius: '4px',
    border: '1px solid #3e4451',
    marginBottom: '16px',
    overflow: 'hidden'
  },
  '.code-header': {
    backgroundColor: '#21252b',
    padding: '6px 12px',
    color: '#abb2bf',
    borderBottom: '1px solid #3e4451',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '12px'
  },
};

// LineageGraph component for visualizing model and column lineage
const LineageGraph = ({ data }) => {
  // Default to sample data if none provided
  const graphData = data || {
    models: [
      { id: 'model1', name: 'order_data', path: 'models/staging/order_data.sql', type: 'staging', highlight: true },
      { id: 'model2', name: 'order_items', path: 'models/intermediate/order_items.sql', type: 'intermediate', highlight: false },
      { id: 'model3', name: 'financial_summary', path: 'models/marts/financial_summary.sql', type: 'mart', highlight: false },
      { id: 'model4', name: 'revenue_projection', path: 'models/marts/revenue_projection.sql', type: 'mart', highlight: false }
    ],
    columns: [
      { id: 'col1', modelId: 'model1', name: 'order_id', type: 'primary_key' },
      { id: 'col2', modelId: 'model1', name: 'customer_id', type: 'foreign_key' },
      { id: 'col3', modelId: 'model1', name: 'order_date', type: 'regular' },
      { id: 'col4', modelId: 'model1', name: 'total_amount', type: 'regular' },
      
      { id: 'col5', modelId: 'model2', name: 'order_id', type: 'foreign_key', sourceId: 'col1' },
      { id: 'col6', modelId: 'model2', name: 'item_id', type: 'primary_key' },
      { id: 'col7', modelId: 'model2', name: 'quantity', type: 'regular' },
      { id: 'col8', modelId: 'model2', name: 'item_price', type: 'regular' },
      { id: 'col9', modelId: 'model2', name: 'line_total', type: 'calculated' },
      
      { id: 'col10', modelId: 'model3', name: 'order_id', type: 'foreign_key', sourceId: 'col1' },
      { id: 'col11', modelId: 'model3', name: 'customer_id', type: 'foreign_key', sourceId: 'col2' },
      { id: 'col12', modelId: 'model3', name: 'total_revenue', type: 'calculated' },
      { id: 'col13', modelId: 'model3', name: 'item_count', type: 'calculated' },
      
      { id: 'col14', modelId: 'model4', name: 'monthly_revenue', type: 'derived', sourceId: 'col12' },
      { id: 'col15', modelId: 'model4', name: 'forecast_next_month', type: 'calculated' }
    ],
    edges: [
      { source: 'model1', target: 'model2' },
      { source: 'model1', target: 'model3' },
      { source: 'model2', target: 'model3' },
      { source: 'model3', target: 'model4' }
    ]
  };

  // Function to determine which models should be expanded initially
  const getInitialExpandedModels = () => {
    const result = {};
    
    // Find the highlighted model
    const highlightedModel = graphData.models.find(m => m.highlight);
    if (highlightedModel) {
      result[highlightedModel.id] = true;
      
      // Get models directly connected to the highlighted model
      const connectedModels = graphData.edges
        .filter(e => e.source === highlightedModel.id || e.target === highlightedModel.id)
        .map(e => e.source === highlightedModel.id ? e.target : e.source);
        
      // Expand those models too
      connectedModels.forEach(modelId => {
        result[modelId] = true;
      });
    } else if (graphData.models.length > 0) {
      // If no highlight, expand the first model
      result[graphData.models[0].id] = true;
    }
    
    return result;
  };
  
  // State for tracking expanded models, active edge, and hover state
  const [expandedModels, setExpandedModels] = useState(getInitialExpandedModels());
  const [activeEdge, setActiveEdge] = useState(null);
  const [activeModelId, setActiveModelId] = useState(null);
  const [activeColumnLink, setActiveColumnLink] = useState(null);

  // Function to toggle model expansion
  const toggleModelExpand = (modelId) => {
    setExpandedModels(prev => ({
      ...prev,
      [modelId]: !prev[modelId]
    }));
  };

  // Function to get the color for a model based on its type
  const getModelTypeColor = (type) => {
    switch (type) {
      case 'staging':
        return 'blue.500';
      case 'intermediate':
        return 'purple.500';
      case 'mart':
        return 'green.500';
      default:
        return 'gray.500';
    }
  };
  
  // Function to get the color for a column based on its type
  const getColumnTypeColor = (type) => {
    switch (type) {
      case 'primary_key':
        return 'yellow.400';
      case 'foreign_key':
        return 'orange.400';
      case 'calculated':
        return 'teal.400';
      case 'derived':
        return 'cyan.400';
      default:
        return 'gray.400';
    }
  };
  
  // Calculate layout positions
  const layout = useMemo(() => {
    const result = {};
    const levels = {};
    const visited = {};
    
    // Function to calculate levels for each model
    const calculateLevels = (modelId, level = 0) => {
      if (visited[modelId]) return;
      visited[modelId] = true;
      
      // Update the level if this one is deeper
      levels[modelId] = Math.max(level, levels[modelId] || 0);
      
      // Process outgoing edges
      graphData.edges
        .filter(e => e.source === modelId)
        .forEach(e => calculateLevels(e.target, level + 1));
    };
    
    // Find source models (no incoming edges)
    const sourceModels = graphData.models
      .filter(m => !graphData.edges.some(e => e.target === m.id))
      .map(m => m.id);
    
    if (sourceModels.length === 0 && graphData.models.length > 0) {
      // If no source models, use the highlighted model or first model
      const startModel = graphData.models.find(m => m.highlight) || graphData.models[0];
      calculateLevels(startModel.id);
    } else {
      // Calculate levels starting from source models
      sourceModels.forEach(id => calculateLevels(id));
    }
    
    // Handle disconnected components
    graphData.models.forEach(model => {
      if (!visited[model.id]) {
        calculateLevels(model.id);
      }
    });
    
    // Group models by level
    const modelsByLevel = {};
    Object.entries(levels).forEach(([modelId, level]) => {
      if (!modelsByLevel[level]) modelsByLevel[level] = [];
      modelsByLevel[level].push(modelId);
    });
    
    // Position models
    Object.entries(modelsByLevel).forEach(([level, modelIds]) => {
      const numModels = modelIds.length;
      modelIds.forEach((modelId, index) => {
        result[modelId] = { x: level * 300 + 50, y: (index - numModels / 2) * 180 + 250 };
      });
    });
    
    return result;
  }, [graphData]);
  
  // Render the connections between models
  const renderEdges = () => {
    return graphData.edges.map((edge, index) => {
      const source = layout[edge.source];
      const target = layout[edge.target];
      
      if (!source || !target) return null;
      
      const isActive = activeEdge === index;
      
      // Calculate path
      const path = `M ${source.x + 125} ${source.y} C ${source.x + 200} ${source.y}, ${target.x + 50} ${target.y}, ${target.x} ${target.y}`;
      
      // Generate a unique animation key
      const animationKey = `edge-${edge.source}-${edge.target}`;
      
      return (
        <Box 
          key={index} 
          position="absolute" 
          zIndex={1}
          onMouseEnter={() => setActiveEdge(index)}
          onMouseLeave={() => setActiveEdge(null)}
        >
          <svg width="100%" height="100%" style={{ position: 'absolute', pointerEvents: 'none' }}>
            <defs>
              <linearGradient id={`gradient-${index}`} x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={getModelTypeColor(graphData.models.find(m => m.id === edge.source)?.type)} />
                <stop offset="100%" stopColor={getModelTypeColor(graphData.models.find(m => m.id === edge.target)?.type)} />
              </linearGradient>
              {isActive && (
                <filter id={`glow-${index}`} x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              )}
            </defs>
            <path 
              d={path} 
              fill="none" 
              stroke={`url(#gradient-${index})`} 
              strokeWidth={isActive ? 4 : 2} 
              strokeDasharray={isActive ? "5,5" : "none"}
              filter={isActive ? `url(#glow-${index})` : "none"}
              opacity={isActive ? 1 : 0.7}
            >
              {isActive && (
                <animate 
                  attributeName="stroke-dashoffset" 
                  values="0;100" 
                  dur="1.5s" 
                  repeatCount="indefinite" 
                />
              )}
            </path>
            <circle 
              cx={target.x} 
              cy={target.y} 
              r={isActive ? 8 : 5} 
              fill={getModelTypeColor(graphData.models.find(m => m.id === edge.target)?.type)} 
              opacity={isActive ? 1 : 0.7}
            >
              {isActive && (
                <animate 
                  attributeName="r" 
                  values="5;8;5" 
                  dur="1s" 
                  repeatCount="indefinite" 
                />
              )}
            </circle>
          </svg>
        </Box>
      );
    });
  };
  
  // Render column connections
  const renderColumnEdges = () => {
    return graphData.columns
      .filter(col => col.sourceId && expandedModels[col.modelId])
      .map((col, index) => {
        const sourceColumn = graphData.columns.find(c => c.id === col.sourceId);
        if (!sourceColumn || !expandedModels[sourceColumn.modelId]) return null;
        
        const sourceModel = graphData.models.find(m => m.id === sourceColumn.modelId);
        const targetModel = graphData.models.find(m => m.id === col.modelId);
        
        if (!sourceModel || !targetModel) return null;
        
        const sourcePos = layout[sourceModel.id];
        const targetPos = layout[targetModel.id];
        
        if (!sourcePos || !targetPos) return null;
        
        // Find index of column in model's column list to calculate y offset
        const sourceColumns = graphData.columns.filter(c => c.modelId === sourceModel.id);
        const targetColumns = graphData.columns.filter(c => c.modelId === col.modelId);
        
        const sourceIndex = sourceColumns.findIndex(c => c.id === sourceColumn.id);
        const targetIndex = targetColumns.findIndex(c => c.id === col.id);
        
        const sourceYOffset = 70 + sourceIndex * 30; // 70px header + 30px per column
        const targetYOffset = 70 + targetIndex * 30;
        
        const isActive = activeColumnLink === `${sourceColumn.id}-${col.id}`;
        
        // Calculate path from source column to target column
        const path = `M ${sourcePos.x + 230} ${sourcePos.y + sourceYOffset} 
                     C ${sourcePos.x + 280} ${sourcePos.y + sourceYOffset}, 
                       ${targetPos.x - 30} ${targetPos.y + targetYOffset}, 
                       ${targetPos.x + 20} ${targetPos.y + targetYOffset}`;
        
        return (
          <Box 
            key={`col-${index}`} 
            position="absolute" 
            zIndex={1}
            onMouseEnter={() => setActiveColumnLink(`${sourceColumn.id}-${col.id}`)}
            onMouseLeave={() => setActiveColumnLink(null)}
          >
            <svg width="100%" height="100%" style={{ position: 'absolute', pointerEvents: 'none' }}>
              <defs>
                <linearGradient id={`col-gradient-${index}`} x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor={getColumnTypeColor(sourceColumn.type)} />
                  <stop offset="100%" stopColor={getColumnTypeColor(col.type)} />
                </linearGradient>
                {isActive && (
                  <filter id={`col-glow-${index}`} x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                )}
              </defs>
              <path 
                d={path} 
                fill="none" 
                stroke={`url(#col-gradient-${index})`} 
                strokeWidth={isActive ? 3 : 1.5} 
                strokeDasharray={isActive ? "3,3" : "none"}
                filter={isActive ? `url(#col-glow-${index})` : "none"}
                opacity={isActive ? 1 : 0.6}
              >
                {isActive && (
                  <animate 
                    attributeName="stroke-dashoffset" 
                    values="0;30" 
                    dur="1s" 
                    repeatCount="indefinite" 
                  />
                )}
              </path>
              <circle 
                cx={targetPos.x + 20} 
                cy={targetPos.y + targetYOffset} 
                r={isActive ? 6 : 4} 
                fill={getColumnTypeColor(col.type)} 
                opacity={isActive ? 1 : 0.8}
              >
                {isActive && (
                  <animate 
                    attributeName="r" 
                    values="4;6;4" 
                    dur="1s" 
                    repeatCount="indefinite" 
                  />
                )}
              </circle>
            </svg>
          </Box>
        );
      });
  };
  
  // Render the models
  const renderModels = () => {
    return graphData.models.map(model => {
      const pos = layout[model.id];
      if (!pos) return null;
      
      const modelColumns = graphData.columns.filter(c => c.modelId === model.id);
      const isExpanded = !!expandedModels[model.id];
      const isActive = activeModelId === model.id;
      
      // Calculate height based on expansion state and number of columns
      const height = isExpanded ? 80 + modelColumns.length * 30 : 80;
      
      return (
        <Box
          key={model.id}
          position="absolute"
          left={`${pos.x}px`}
          top={`${pos.y}px`}
          width="250px"
          height={`${height}px`}
          bg={model.highlight ? "gray.700" : "gray.800"}
          borderColor={isActive || model.highlight ? getModelTypeColor(model.type) : "gray.700"}
          borderWidth="2px"
          borderRadius="md"
          boxShadow={isActive || model.highlight ? `0 0 10px ${getModelTypeColor(model.type)}` : "md"}
          transition="all 0.3s ease, height 0.3s ease, box-shadow 0.3s ease"
          _hover={{ boxShadow: `0 0 15px ${getModelTypeColor(model.type)}` }}
          onMouseEnter={() => setActiveModelId(model.id)}
          onMouseLeave={() => setActiveModelId(null)}
          overflow="hidden"
        >
          <HStack p={2} bg={`${getModelTypeColor(model.type)}22`} justifyContent="space-between">
            <HStack>
              <Box w="10px" h="10px" borderRadius="full" bg={getModelTypeColor(model.type)} />
              <Text fontWeight="bold" isTruncated maxW="150px">{model.name}</Text>
            </HStack>
            <HStack spacing={2}>
              <Tag size="sm" colorScheme={model.type === 'staging' ? 'blue' : model.type === 'intermediate' ? 'purple' : 'green'}>
                {model.type}
              </Tag>
              <IconButton 
                aria-label={isExpanded ? "Collapse" : "Expand"}
                icon={isExpanded ? <IoChevronUp /> : <IoChevronDown />}
                size="xs"
                variant="ghost"
                onClick={() => toggleModelExpand(model.id)}
              />
            </HStack>
          </HStack>
          <Text fontSize="xs" color="gray.400" px={3} pt={1} isTruncated>
            {model.path}
          </Text>
          
          {isExpanded && (
            <VStack align="start" p={2} spacing={1} mt={1}>
              {modelColumns.map(column => (
                <HStack key={column.id} w="100%" spacing={2}>
                  <Box w="8px" h="8px" borderRadius="full" bg={getColumnTypeColor(column.type)} mt="2px" />
                  <Text fontSize="sm" isTruncated maxW="150px">{column.name}</Text>
                  <Tag size="sm" ml="auto" colorScheme={
                    column.type === 'primary_key' ? 'yellow' : 
                    column.type === 'foreign_key' ? 'orange' : 
                    column.type === 'calculated' ? 'teal' : 
                    column.type === 'derived' ? 'cyan' : 'gray'
                  }>
                    <TagLabel fontSize="xs">{column.type.replace('_', ' ')}</TagLabel>
                  </Tag>
                </HStack>
              ))}
            </VStack>
          )}
        </Box>
      );
    });
  };
  
  // Render a legend for the visualization
  const renderLegend = () => {
    return (
      <Box 
        position="absolute" 
        top="20px" 
        right="20px" 
        bg="gray.800" 
        p={3} 
        borderRadius="md" 
        boxShadow="md"
        maxW="220px"
      >
        <Text fontWeight="bold" mb={2}>Legend</Text>
        
        <Text fontSize="sm" fontWeight="medium" mb={1}>Model Types</Text>
        <HStack mb={2}>
          <Box w="10px" h="10px" borderRadius="full" bg="blue.500" />
          <Text fontSize="xs">Staging</Text>
          <Box w="10px" h="10px" borderRadius="full" bg="purple.500" ml={2} />
          <Text fontSize="xs">Intermediate</Text>
          <Box w="10px" h="10px" borderRadius="full" bg="green.500" ml={2} />
          <Text fontSize="xs">Mart</Text>
        </HStack>
        
        <Text fontSize="sm" fontWeight="medium" mb={1}>Column Types</Text>
        <HStack mb={1}>
          <Box w="10px" h="10px" borderRadius="full" bg="yellow.400" />
          <Text fontSize="xs">Primary Key</Text>
          <Box w="10px" h="10px" borderRadius="full" bg="orange.400" ml={2} />
          <Text fontSize="xs">Foreign Key</Text>
        </HStack>
        <HStack>
          <Box w="10px" h="10px" borderRadius="full" bg="teal.400" />
          <Text fontSize="xs">Calculated</Text>
          <Box w="10px" h="10px" borderRadius="full" bg="cyan.400" ml={2} />
          <Text fontSize="xs">Derived</Text>
          <Box w="10px" h="10px" borderRadius="full" bg="gray.400" ml={2} />
          <Text fontSize="xs">Regular</Text>
        </HStack>
      </Box>
    );
  };
  
  return (
    <Box
      width="100%"
      height="600px"
      position="relative"
      bg="gray.900"
      borderRadius="md"
      overflow="hidden"
      mt={4}
      mb={4}
    >
      {/* Render background grid */}
      <Box 
        position="absolute" 
        top="0" 
        left="0" 
        right="0" 
        bottom="0"
        backgroundImage="linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px)"
        backgroundSize="20px 20px"
        opacity="0.5"
      />
      
      {renderEdges()}
      {renderColumnEdges()}
      {renderModels()}
      {renderLegend()}
      
      <Text position="absolute" bottom="10px" right="10px" fontSize="xs" color="gray.500">
        Interactive Data Lineage Visualization
      </Text>
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
      
      // Look for column definitions when showing detailed model info
      if (line.includes('[') && line.includes(']')) {
        // Find the current model context
        let modelContext = null;
        
        // Look back a few lines to find the model context
        for (let j = i - 1; j >= Math.max(0, i - 5); j--) {
          const contextLine = lines[j].trim();
          const contextModel = data.models.find(model => 
            contextLine.includes(model.path) || 
            contextLine.includes(`└── ${model.name}`) ||
            contextLine.includes(`── ${model.name}`));
          
          if (contextModel) {
            modelContext = contextModel;
            break;
          }
        }
        
        if (modelContext) {
          // Extract column info
          const columnMatch = line.match(/\[([^\]]+)\]\s*\(([^)]+)\)/);
          if (columnMatch) {
            const columnName = columnMatch[1].trim();
            const columnDesc = columnMatch[2].trim();
            
            // Determine column type based on description
            let columnType = 'regular';
            
            if (columnDesc.toLowerCase().includes('primary key')) {
              columnType = 'primary_key';
            } else if (columnDesc.toLowerCase().includes('foreign key')) {
              columnType = 'foreign_key';
            } else if (columnDesc.toLowerCase().includes('calculated') || 
                      columnDesc.toLowerCase().includes('derived')) {
              columnType = 'calculated';
            }
            
            // Add column
            const columnId = `col${++columnCount}`;
            data.columns.push({
              id: columnId,
              modelId: modelContext.id,
              name: columnName,
              type: columnType
            });
            
            // Check for reference to other columns
            for (const model of data.models) {
              if (columnDesc.includes(model.name)) {
                // This column might reference a column in another model
                const modelColumns = data.columns.filter(c => c.modelId === model.id);
                for (const sourceColumn of modelColumns) {
                  if (columnDesc.includes(sourceColumn.name)) {
                    // Found a reference
                    const currentColumn = data.columns.find(c => c.id === columnId);
                    if (currentColumn) {
                      currentColumn.sourceId = sourceColumn.id;
                      break;
                    }
                  }
                }
              }
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

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const messagesEndRef = useRef(null);
  const [currentConversationId, setCurrentConversationId] = useState(conversationId || null);
  
  const bgColor = useColorModeValue('white', 'gray.900')
  const textColor = useColorModeValue('gray.900', 'white')
  const borderColor = useColorModeValue('gray.100', 'gray.700')
  const primaryColor = 'orange.500'

  // Apply confluence styles
  React.useEffect(() => {
    // Create style element
    const styleEl = document.createElement('style');
    let cssText = '';
    
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

  // Initialize or load conversation from ID
  useEffect(() => {
    // Clear messages when component mounts if there's no conversation ID
    if (!conversationId) {
      setMessages([]);
      setActiveConversationId(null);
      setCurrentConversationId(null);
    } else {
      // If we have a conversation ID on mount, fetch that conversation
      fetchConversation(conversationId);
    }
  }, [conversationId]);

  // Fetch conversation history
  const fetchConversations = async () => {
    try {
      setIsHistoryLoading(true);
      const response = await fetch('http://localhost:8000/api/conversations');
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }
      const data = await response.json();
      if (data.status === 'success') {
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      toast({
        title: 'Error',
        description: 'Failed to load conversation history',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setIsHistoryLoading(false);
    }
  };

  // Fetch a specific conversation
  const fetchConversation = async (id) => {
    try {
      setProcessingStep("Loading conversation...");
      console.log("Fetching conversation with ID:", id);
      
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching conversation: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation data:", data);
      
      if (!data.conversation) {
        throw new Error("No conversation data received");
      }
      
      // Extract conversation data
      const conversation = data.conversation;
      
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
        formattedMessages.push({
          id: `assistant-${id}`,
          role: 'assistant',
          type: 'architect',
          content: conversation.response,
          details: {
            ...conversation.technical_details,
            conversation_id: id
          },
          timestamp: conversation.timestamp
        });
      }
      
      // Update state
      setMessages(formattedMessages);
      setActiveConversationId(id);
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
      setProcessingStep(null);
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Start a new conversation
  const startNewConversation = () => {
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    navigate('/chat', { replace: true });
  };

  // Handle conversation selection
  const handleConversationSelect = (id) => {
    fetchConversation(id);
    onClose(); // Close the drawer on mobile
  };

  // Send message and get response from data architect
  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = input;
    setInput('');
    
    // Add user message to the chat
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: userMessage,
      id: prev.length
    }]);
    
    // Set loading state
    setLoading(true);
    setProcessingStep('Analyzing your question with Data Architect...');
    
    try {
      // Use the Data Architect agent endpoint
      const response = await fetch('http://localhost:8000/architect/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          query: userMessage,
          conversation_id: currentConversationId,
          thread_id: currentConversationId
        })
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Data Architect response:", data);
      
      // Create a conversation ID if needed
      const responseConversationId = data.conversation_id || currentConversationId || uuidv4();
      
      // Add the Data Architect's response to the chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        type: 'architect',
        content: data.response,
        id: prev.length,
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
      }]);
      
      // Update conversation tracking info
      setCurrentConversationId(responseConversationId);
      
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: 'Error',
        description: 'Failed to send message. Please try again.',
        status: 'error',
        duration: 3000,
      });
      
      // Add error message to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        id: prev.length,
        isError: true
      }]);
    } finally {
      setLoading(false);
      setProcessingStep('');
    }
  };

  // Clear the current chat
  const clearChat = () => {
    // Save current conversation ID before clearing
    const currentId = activeConversationId;
    
    // Clear messages and conversation ID
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    
    toast({
      title: 'Chat Cleared',
      description: 'The chat has been cleared. You can find it in the history tab.',
      status: 'info',
      duration: 3000
    });
  };

  // Handle formatted responses from different agents
  const handleFormattedResponses = (response) => {
    if (response.agent === 'architect') {
      const sections = {};
      
      // Process each section
      if (response.sections) {
        Object.keys(response.sections).forEach(key => {
          sections[key] = response.sections[key];
        });
      }
      
      // Process implementation details
      if (response.implementation) {
        sections.implementation = response.implementation;
      }
      
      // Combine all sections into a single content string
      let content = '';
      Object.entries(sections).forEach(([key, value]) => {
        if (typeof value === 'string') {
          const formattedKey = key.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          content += `## ${formattedKey}\n\n${value}\n\n`;
        }
      });
      
      return renderArchitectResponse(content, sections);
    }
    
    // Handle other agent types as needed
    return <Text>{JSON.stringify(response)}</Text>;
  };

  return (
    <Box minH="100vh" bg={bgColor}>
      <Container maxW="container.lg" py={8}>
        {messages.length === 0 ? (
          // Initial empty state
          <VStack spacing={8} align="center" textAlign="center" py={20}>
            <Heading 
              size="xl" 
              color={textColor}
              lineHeight="1.2"
            >
              Data Architect Assistant
            </Heading>

            <Text fontSize="lg" color="gray.600" maxW="600px">
              Ask questions about your database schemas, data models, or get recommendations for SQL optimization
            </Text>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} pt={8} w="full">
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoServer} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">Schema Design</Text>
                <Text fontSize="sm" color="gray.600">
                  "Help me optimize my database schema"
                </Text>
              </VStack>
              
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoCodeSlash} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">dbt Models</Text>
                <Text fontSize="sm" color="gray.600">
                  "Review my dbt model structure"
                </Text>
              </VStack>
              
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoAnalytics} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">Query Analysis</Text>
                <Text fontSize="sm" color="gray.600">
                  "Optimize this SQL query"
                </Text>
              </VStack>
            </SimpleGrid>
          </VStack>
        ) : (
          // Chat messages area
          <VStack spacing={4} h="full">
            {messages.map((message) => renderMessage(message))}
            
            {processingStep && (
              <Box 
                p={4} 
                bg="blue.50" 
                borderRadius="md" 
                width={["98%", "95%", "90%"]}
                alignSelf="flex-start"
                boxShadow="0 2px 8px rgba(0, 0, 0, 0.05)"
                borderWidth="1px"
                borderColor="blue.100"
              >
                <HStack>
                  <Box as={IoAnalytics} color="blue.500" boxSize={5} mr={2} />
                  <Text fontWeight="500" color="blue.700">{processingStep}</Text>
                </HStack>
                <Progress size="xs" colorScheme="blue" isIndeterminate mt={3} />
              </Box>
            )}
            <div ref={messagesEndRef} />
          </VStack>
        )}

        {/* Input Area */}
        <Box 
          position="fixed"
          bottom={0}
          left={0}
          right={0}
          p={4}
          bg={bgColor}
          borderTop="1px"
          borderColor={borderColor}
          zIndex={2}
        >
          <Container maxW="container.md">
            <HStack spacing={4}>
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about database schemas, data models, or SQL queries..."
                size="lg"
                bg={useColorModeValue('white', 'gray.800')}
                borderColor={borderColor}
                _focus={{
                  borderColor: primaryColor,
                  boxShadow: `0 0 0 1px ${primaryColor}`
                }}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    sendMessage();
                  }
                }}
              />
              <Button
                colorScheme="orange"
                size="lg"
                px={8}
                isLoading={loading}
                onClick={sendMessage}
                leftIcon={<IoSend />}
              >
                Send
              </Button>
            </HStack>
          </Container>
        </Box>
      </Container>
    </Box>
  );
};

export default ChatPage; 