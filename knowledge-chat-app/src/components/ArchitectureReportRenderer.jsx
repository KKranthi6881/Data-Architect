import React from 'react';
import {
  Box,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Text,
  Heading,
  Flex,
  Divider,
  Icon,
  HStack,
  VStack,
  useColorModeValue,
  Code
} from '@chakra-ui/react';
import { 
  IoAnalytics, 
  IoServer, 
  IoCode,
  IoCheckmarkCircle,
  IoLayers,
  IoBusiness
} from 'react-icons/io5';

const ArchitectureReportRenderer = ({ content }) => {
  const headerBg = useColorModeValue('purple.50', 'purple.900');
  const sectionBg = useColorModeValue('blue.50', 'blue.900');
  const tableBorderColor = useColorModeValue('purple.200', 'purple.700');
  const altRowBg = useColorModeValue('gray.50', 'gray.700');
  const codeBg = useColorModeValue('gray.50', 'gray.800');
  
  if (!content) return null;
  
  // Parse the report into sections
  const parseReportSections = (text) => {
    const sections = [];
    const sectionHeaderRegex = /## ([^\n]+)/g;
    let lastIndex = 0;
    let match;
    
    while ((match = sectionHeaderRegex.exec(text)) !== null) {
      const title = match[1];
      const startPos = match.index;
      
      // If this isn't the first match, extract the content of the previous section
      if (lastIndex > 0) {
        const content = text.substring(lastIndex, startPos).trim();
        sections.push({
          title: sections[sections.length - 1].title,
          content: content
        });
      }
      
      // Add this section header
      sections.push({
        title: title,
        content: ''
      });
      
      lastIndex = startPos + match[0].length;
    }
    
    // Add the final section content
    if (lastIndex > 0 && lastIndex < text.length) {
      const content = text.substring(lastIndex).trim();
      sections.push({
        title: sections[sections.length - 1].title,
        content: content
      });
      // Remove the empty header entry
      sections.splice(sections.length - 2, 1);
    }
    
    return sections;
  };
  
  // Helper function to render tables inside a section
  const renderTables = (text) => {
    // Find all tables in the text
    const tableRegex = /### ([^\n]+)\n\n\| (.*) \|\n\| [-:|]+ \|\n((.*\n)+?)(?=\n###|\n##|$)/g;
    const tables = [];
    let tableMatch;
    
    while ((tableMatch = tableRegex.exec(text)) !== null) {
      const tableName = tableMatch[1];
      const headerLine = tableMatch[2];
      const bodyContent = tableMatch[3];
      
      // Parse headers
      const headers = headerLine.split('|').map(h => h.trim()).filter(h => h);
      
      // Parse rows
      const rows = [];
      const rowLines = bodyContent.trim().split('\n');
      
      for (const line of rowLines) {
        if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
          const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell);
          if (cells.length === headers.length) {
            rows.push(cells);
          }
        }
      }
      
      tables.push({
        name: tableName,
        headers,
        rows
      });
    }
    
    return tables;
  };
  
  // Helper function to extract code blocks
  const extractCodeBlocks = (text) => {
    const codeRegex = /```([a-z]*)\n([\s\S]*?)```/g;
    const codeBlocks = [];
    let codeMatch;
    
    while ((codeMatch = codeRegex.exec(text)) !== null) {
      codeBlocks.push({
        language: codeMatch[1] || 'text',
        code: codeMatch[2]
      });
    }
    
    return codeBlocks;
  };
  
  // Parse the sections
  const sections = parseReportSections(content);
  
  // Helper function to get the section icon
  const getSectionIcon = (title) => {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('business')) return IoBusiness;
    if (lowerTitle.includes('architecture')) return IoServer;
    if (lowerTitle.includes('schema')) return IoLayers;
    if (lowerTitle.includes('implementation')) return IoAnalytics;
    if (lowerTitle.includes('code')) return IoCode;
    if (lowerTitle.includes('test')) return IoCheckmarkCircle;
    return IoAnalytics;
  };
  
  return (
    <Box>
      {sections.map((section, index) => {
        // Skip empty sections or header-only sections
        if (!section.content && index % 2 === 0) return null;
        
        const SectionIcon = getSectionIcon(section.title);
        
        // For section headers (odd indices)
        if (index % 2 === 0) {
          return (
            <Box 
              key={`header-${index}`}
              mt={index === 0 ? 0 : 6} 
              mb={3}
              pt={index === 0 ? 0 : 4}
              borderTopWidth={index === 0 ? 0 : "1px"}
              borderColor="gray.200"
            >
              <HStack spacing={2} align="center">
                <Icon as={SectionIcon} color="purple.500" boxSize={6} />
                <Heading as="h2" size="lg" color="purple.700">
                  {section.title}
                </Heading>
              </HStack>
              <Divider mt={2} borderColor="purple.200" />
            </Box>
          );
        }
        
        // For section content (even indices)
        // Check if this section contains tables
        const tables = renderTables(section.content);
        const codeBlocks = extractCodeBlocks(section.content);
        
        // Remove table and code content from the text to avoid duplication
        let cleanedContent = section.content;
        
        // Remove table content
        tables.forEach(table => {
          const tablePattern = new RegExp(`### ${table.name}[\\s\\S]*?\\n\\|[\\s\\S]*?(?=\\n###|\\n##|$)`, 'g');
          cleanedContent = cleanedContent.replace(tablePattern, '');
        });
        
        // Remove code block content
        codeBlocks.forEach(codeBlock => {
          const codePattern = new RegExp(`\`\`\`${codeBlock.language}\\n${codeBlock.code.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\`\`\``, 'g');
          cleanedContent = cleanedContent.replace(codePattern, '');
        });
        
        // Process remaining text to handle paragraphs and lists
        const paragraphs = cleanedContent.split('\n\n').filter(p => p.trim());
        
        return (
          <Box key={`content-${index}`} mx={2}>
            {/* Regular text content */}
            {paragraphs.map((paragraph, pIdx) => {
              // Check if paragraph is a list
              if (paragraph.trim().match(/^\d+\.|^- /m)) {
                const items = paragraph.split('\n').filter(item => item.trim());
                const isNumbered = items[0].trim().match(/^\d+\./);
                
                return (
                  <Box key={`list-${pIdx}`} my={3} ml={4}>
                    {items.map((item, itemIdx) => (
                      <HStack key={`item-${itemIdx}`} align="start" spacing={2} mb={2}>
                        <Text fontWeight="bold" color="purple.500" minW="24px">
                          {isNumbered ? item.trim().match(/^\d+/)[0] + '.' : '•'}
                        </Text>
                        <Text>{item.replace(/^\d+\.\s*|- /, '')}</Text>
                      </HStack>
                    ))}
                  </Box>
                );
              }
              
              // Regular paragraph
              return (
                <Text key={`para-${pIdx}`} my={3} lineHeight="tall">
                  {paragraph}
                </Text>
              );
            })}
            
            {/* Tables */}
            {tables.map((table, tIdx) => (
              <Box 
                key={`table-${tIdx}`} 
                my={5}
                border="1px solid" 
                borderColor={tableBorderColor} 
                borderRadius="md" 
                overflow="hidden"
              >
                <Flex 
                  bg={headerBg} 
                  p={3}
                  alignItems="center"
                  borderBottom="1px solid"
                  borderColor={tableBorderColor}
                >
                  <Heading size="md" color="purple.700">
                    {table.name}
                  </Heading>
                </Flex>
                
                <Box overflowX="auto">
                  <Table variant="simple" size="sm">
                    <Thead bg={headerBg}>
                      <Tr>
                        {table.headers.map((header, hIdx) => (
                          <Th key={`header-${hIdx}`}>{header}</Th>
                        ))}
                      </Tr>
                    </Thead>
                    <Tbody>
                      {table.rows.map((row, rIdx) => (
                        <Tr key={`row-${rIdx}`} bg={rIdx % 2 === 1 ? altRowBg : 'transparent'}>
                          {row.map((cell, cIdx) => (
                            <Td key={`cell-${cIdx}`}>
                              {cIdx === 0 ? (
                                <Text fontWeight="medium">{cell}</Text>
                              ) : (
                                <Text>{cell}</Text>
                              )}
                            </Td>
                          ))}
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </Box>
              </Box>
            ))}
            
            {/* Code blocks */}
            {codeBlocks.map((codeBlock, cIdx) => (
              <Box key={`code-${cIdx}`} my={5}>
                <HStack mb={2}>
                  <Icon as={IoCode} color="blue.500" />
                  <Text color="blue.500" fontWeight="medium">
                    {codeBlock.language === 'python' ? 'Python Code' : 
                     codeBlock.language === 'sql' ? 'SQL Query' : 
                     'Code Example'}
                  </Text>
                </HStack>
                <Code
                  display="block"
                  whiteSpace="pre"
                  p={4}
                  borderRadius="md"
                  bg={codeBg}
                  overflowX="auto"
                  fontSize="sm"
                  width="100%"
                >
                  {codeBlock.code}
                </Code>
              </Box>
            ))}
          </Box>
        );
      })}
    </Box>
  );
};

export default ArchitectureReportRenderer; 