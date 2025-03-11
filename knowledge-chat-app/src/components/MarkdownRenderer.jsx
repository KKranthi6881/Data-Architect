import React, { useEffect, useState } from 'react';
import {
  Box,
  Text,
  Heading,
  Code,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  OrderedList,
  UnorderedList,
  ListItem,
  Divider,
  useColorModeValue
} from '@chakra-ui/react';
import SchemaRenderer from './SchemaRenderer';

const MarkdownRenderer = ({ content }) => {
  const [schemaData, setSchemaData] = useState(null);
  const codeBg = useColorModeValue('gray.50', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Extract schema information from the content
  useEffect(() => {
    if (!content) return;
    
    // Look for schema sections in the content
    const schemaRegex = /## (Data Architecture|Schema Design)([\s\S]*?)(?=##|$)/;
    const schemaMatch = content.match(schemaRegex);
    
    if (schemaMatch && schemaMatch[2]) {
      setSchemaData(schemaMatch[2].trim());
    } else {
      setSchemaData(null);
    }
  }, [content]);
  
  if (!content) return null;
  
  // First, parse and replace markdown tables
  const renderTables = (text) => {
    // Find table patterns
    const tableRegex = /\|(.+)\|\n\|([\s-:]+)\|\n((?:\|.+\|\n)+)/g;
    
    return text.replace(tableRegex, (match, headerRow, separatorRow, bodyRows) => {
      // Parse header columns
      const headers = headerRow.split('|').map(h => h.trim()).filter(h => h);
      
      // Parse body rows
      const rows = bodyRows
        .trim()
        .split('\n')
        .map(row => 
          row
            .split('|')
            .map(cell => cell.trim())
            .filter((cell, index) => index > 0 && index <= headers.length)
        );
      
      // Return a placeholder that we'll replace with JSX later
      return `__TABLE_PLACEHOLDER_${JSON.stringify({ headers, rows })}_END_TABLE__`;
    });
  };
  
  // Parse and replace code blocks
  const renderCodeBlocks = (text) => {
    return text.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, language, code) => {
      return `__CODE_PLACEHOLDER_${JSON.stringify({ language, code })}_END_CODE__`;
    });
  };
  
  // Parse and replace headers
  const renderHeaders = (text) => {
    return text
      .replace(/^### (.*$)/gm, '__H3_PLACEHOLDER_$1_END_H3__')
      .replace(/^## (.*$)/gm, '__H2_PLACEHOLDER_$1_END_H2__')
      .replace(/^# (.*$)/gm, '__H1_PLACEHOLDER_$1_END_H1__');
  };
  
  // Parse and replace lists
  const renderLists = (text) => {
    // Ordered lists
    let processedText = text.replace(/^\d+\. (.*)$/gm, '__OL_ITEM_PLACEHOLDER_$1_END_OL_ITEM__');
    
    // Unordered lists
    processedText = processedText.replace(/^[*-] (.*)$/gm, '__UL_ITEM_PLACEHOLDER_$1_END_UL_ITEM__');
    
    return processedText;
  };
  
  // Process the content with all our parsers
  let processedContent = content;
  processedContent = renderTables(processedContent);
  processedContent = renderCodeBlocks(processedContent);
  processedContent = renderHeaders(processedContent);
  processedContent = renderLists(processedContent);
  
  // Split by double newlines to get paragraphs
  const paragraphs = processedContent.split(/\n\n+/);
  
  return (
    <Box>
      {paragraphs.map((paragraph, index) => {
        // Handle heading placeholders
        if (paragraph.startsWith('__H1_PLACEHOLDER_')) {
          const heading = paragraph.replace('__H1_PLACEHOLDER_', '').replace('_END_H1__', '');
          return <Heading key={index} as="h1" size="xl" my={4}>{heading}</Heading>;
        }
        
        if (paragraph.startsWith('__H2_PLACEHOLDER_')) {
          const heading = paragraph.replace('__H2_PLACEHOLDER_', '').replace('_END_H2__', '');
          return <Heading key={index} as="h2" size="lg" my={3} color="blue.600">{heading}</Heading>;
        }
        
        if (paragraph.startsWith('__H3_PLACEHOLDER_')) {
          const heading = paragraph.replace('__H3_PLACEHOLDER_', '').replace('_END_H3__', '');
          return <Heading key={index} as="h3" size="md" my={2} color="blue.500">{heading}</Heading>;
        }
        
        // Handle table placeholders
        if (paragraph.startsWith('__TABLE_PLACEHOLDER_')) {
          const tableJson = paragraph
            .replace('__TABLE_PLACEHOLDER_', '')
            .replace('_END_TABLE__', '');
          
          try {
            const { headers, rows } = JSON.parse(tableJson);
            
            return (
              <Box key={index} my={4} overflowX="auto">
                <Table variant="simple" size="sm" borderWidth="1px" borderColor={borderColor}>
                  <Thead bg={codeBg}>
                    <Tr>
                      {headers.map((header, i) => (
                        <Th key={i}>{header}</Th>
                      ))}
                    </Tr>
                  </Thead>
                  <Tbody>
                    {rows.map((row, rowIndex) => (
                      <Tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <Td key={cellIndex}>{cell}</Td>
                        ))}
                      </Tr>
                    ))}
                  </Tbody>
                </Table>
              </Box>
            );
          } catch (e) {
            return <Text key={index}>{paragraph}</Text>;
          }
        }
        
        // Handle code block placeholders
        if (paragraph.startsWith('__CODE_PLACEHOLDER_')) {
          const codeJson = paragraph
            .replace('__CODE_PLACEHOLDER_', '')
            .replace('_END_CODE__', '');
          
          try {
            const { language, code } = JSON.parse(codeJson);
            
            return (
              <Box key={index} my={4}>
                <Text fontSize="xs" color="gray.500" mb={1}>
                  {language && language.trim() ? language : 'code'}
                </Text>
                <Code
                  display="block"
                  whiteSpace="pre"
                  children={code}
                  p={3}
                  borderRadius="md"
                  bg={codeBg}
                  overflowX="auto"
                  fontSize="sm"
                  width="100%"
                />
              </Box>
            );
          } catch (e) {
            return <Text key={index}>{paragraph}</Text>;
          }
        }
        
        // Handle list items
        if (paragraph.includes('__OL_ITEM_PLACEHOLDER_') || paragraph.includes('__UL_ITEM_PLACEHOLDER_')) {
          const lines = paragraph.split('\n');
          const isOrdered = lines[0].includes('__OL_ITEM_PLACEHOLDER_');
          
          const items = lines.map(line => {
            if (line.includes('__OL_ITEM_PLACEHOLDER_')) {
              return line.replace('__OL_ITEM_PLACEHOLDER_', '').replace('_END_OL_ITEM__', '');
            }
            if (line.includes('__UL_ITEM_PLACEHOLDER_')) {
              return line.replace('__UL_ITEM_PLACEHOLDER_', '').replace('_END_UL_ITEM__', '');
            }
            return line;
          });
          
          const ListComponent = isOrdered ? OrderedList : UnorderedList;
          
          return (
            <ListComponent key={index} spacing={1} pl={4} my={2}>
              {items.map((item, i) => (
                <ListItem key={i}>{item}</ListItem>
              ))}
            </ListComponent>
          );
        }
        
        // Regular paragraph
        return <Text key={index} mb={2}>{paragraph}</Text>;
      })}
      
      {/* Render schema tables if we found any */}
      {schemaData && (
        <Box mt={6}>
          <SchemaRenderer schemaData={schemaData} />
        </Box>
      )}
    </Box>
  );
};

export default MarkdownRenderer; 