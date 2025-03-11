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
  Tooltip,
  Icon,
  HStack,
  VStack,
  useColorModeValue
} from '@chakra-ui/react';
import { IoKey, IoLink, IoInformation, IoLockClosed } from 'react-icons/io5';

const SchemaRenderer = ({ schemaData }) => {
  const headerBg = useColorModeValue('purple.50', 'purple.900');
  const tableBorderColor = useColorModeValue('purple.200', 'purple.700');
  const altRowBg = useColorModeValue('gray.50', 'gray.700');
  
  // If schemaData is a string, attempt to parse it as JSON
  let parsedSchema = schemaData;
  if (typeof schemaData === 'string') {
    try {
      parsedSchema = JSON.parse(schemaData);
    } catch (e) {
      // Not valid JSON, we'll handle as text
      parsedSchema = null;
    }
  }
  
  // Function to extract table info from text if we don't have structured data
  const extractSchemaFromText = (text) => {
    // Look for sections that appear to be schema definitions
    const tableRegex = /## Table: ([^\n]+)\n\n(.*?)(?=\n## |$)/gs;
    const columnRegex = /\| ([^|]+) \| ([^|]+) \| ([^|]+) \|/g;
    
    const tables = [];
    let tableMatch;
    
    while ((tableMatch = tableRegex.exec(text)) !== null) {
      const tableName = tableMatch[1].trim();
      const tableContent = tableMatch[2].trim();
      
      const columns = [];
      let columnMatch;
      
      while ((columnMatch = columnRegex.exec(tableContent)) !== null) {
        columns.push({
          name: columnMatch[1].trim(),
          type: columnMatch[2].trim(),
          description: columnMatch[3].trim()
        });
      }
      
      tables.push({
        name: tableName,
        columns: columns
      });
    }
    
    return tables.length > 0 ? tables : null;
  };
  
  // If we don't have structured data, try to extract it from text
  if (!parsedSchema && typeof schemaData === 'string') {
    parsedSchema = extractSchemaFromText(schemaData);
  }
  
  // If we still don't have structured data, return null
  if (!parsedSchema) {
    return null;
  }
  
  // Normalize to array if it's a single object
  const schemas = Array.isArray(parsedSchema) ? parsedSchema : [parsedSchema];
  
  return (
    <Box>
      {schemas.map((schema, schemaIndex) => (
        <Box 
          key={schemaIndex} 
          mb={6} 
          border="1px solid" 
          borderColor={tableBorderColor} 
          borderRadius="md" 
          overflow="hidden"
        >
          <Flex 
            bg={headerBg} 
            p={3} 
            justifyContent="space-between" 
            alignItems="center"
            borderBottom="1px solid"
            borderColor={tableBorderColor}
          >
            <HStack>
              <Heading size="md" color="purple.700">
                {schema.table_name || schema.name || "Schema Table"}
              </Heading>
              {schema.schema_name && (
                <Badge colorScheme="purple" ml={2}>
                  {schema.schema_name}
                </Badge>
              )}
            </HStack>
            
            {schema.description && (
              <Text fontSize="sm" fontStyle="italic" color="gray.600" ml={4}>
                {schema.description}
              </Text>
            )}
          </Flex>
          
          <Box overflowX="auto">
            <Table variant="simple" size="sm">
              <Thead bg={headerBg}>
                <Tr>
                  <Th width="25%">Column Name</Th>
                  <Th width="20%">Data Type</Th>
                  <Th width="15%">Constraints</Th>
                  <Th>Description</Th>
                </Tr>
              </Thead>
              <Tbody>
                {(schema.columns || []).map((column, idx) => {
                  // Normalize column data structure
                  const col = typeof column === 'string' 
                    ? { name: column, type: '', constraints: [], description: '' }
                    : column;
                  
                  // Extract column name
                  const colName = col.name || col.column_name || '';
                  
                  // Extract column type
                  const colType = col.type || col.data_type || '';
                  
                  // Determine constraints
                  const isPk = 
                    col.is_primary_key || 
                    col.pk || 
                    colName.toLowerCase().includes('_id') || 
                    colName.toLowerCase() === 'id';
                  
                  const isFk = 
                    col.is_foreign_key || 
                    col.fk || 
                    colName.toLowerCase().includes('_id');
                  
                  const isNotNull = 
                    col.is_nullable === false || 
                    col.not_null || 
                    col.nullable === false;
                  
                  const isUnique = 
                    col.is_unique || 
                    col.unique;
                  
                  // Extract description
                  const description = col.description || '';
                  
                  return (
                    <Tr key={idx} bg={idx % 2 === 1 ? altRowBg : 'transparent'}>
                      <Td fontWeight={isPk ? "bold" : "normal"}>
                        <HStack spacing={1} align="center">
                          <Text>{colName}</Text>
                          {isPk && (
                            <Tooltip label="Primary Key">
                              <Badge colorScheme="blue" ml={1}>PK</Badge>
                            </Tooltip>
                          )}
                          {isFk && !isPk && (
                            <Tooltip label="Foreign Key">
                              <Badge colorScheme="green" ml={1}>FK</Badge>
                            </Tooltip>
                          )}
                        </HStack>
                      </Td>
                      <Td>
                        <Badge 
                          colorScheme={
                            colType.includes('int') ? 'blue' :
                            colType.includes('char') || colType.includes('text') ? 'green' :
                            colType.includes('date') || colType.includes('time') ? 'orange' :
                            colType.includes('bool') ? 'red' :
                            'purple'
                          }
                          variant="subtle"
                        >
                          {colType}
                        </Badge>
                      </Td>
                      <Td>
                        <HStack spacing={1} flexWrap="wrap">
                          {isNotNull && (
                            <Tooltip label="Not Null">
                              <Badge colorScheme="red" variant="outline" size="sm">
                                <HStack spacing={1}>
                                  <Icon as={IoLockClosed} boxSize={3} />
                                  <Text fontSize="xs">NOT NULL</Text>
                                </HStack>
                              </Badge>
                            </Tooltip>
                          )}
                          {isUnique && (
                            <Tooltip label="Unique Constraint">
                              <Badge colorScheme="teal" variant="outline" size="sm">UNIQUE</Badge>
                            </Tooltip>
                          )}
                          {col.references && (
                            <Tooltip label={`References ${col.references}`}>
                              <Badge colorScheme="purple" variant="outline" size="sm">
                                <HStack spacing={1}>
                                  <Icon as={IoLink} boxSize={3} />
                                  <Text fontSize="xs">REF</Text>
                                </HStack>
                              </Badge>
                            </Tooltip>
                          )}
                        </HStack>
                      </Td>
                      <Td fontSize="sm">{description}</Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </Box>
          
          {/* Relationships section if available */}
          {schema.relationships && schema.relationships.length > 0 && (
            <Box borderTop="1px solid" borderColor={tableBorderColor} p={3}>
              <Heading size="sm" mb={2} color="purple.600">Relationships</Heading>
              <VStack align="stretch" spacing={2}>
                {schema.relationships.map((rel, relIdx) => (
                  <HStack key={relIdx} spacing={2} bg="gray.50" p={2} borderRadius="md">
                    <Badge colorScheme="blue">{rel.type || 'relates to'}</Badge>
                    <Text fontSize="sm">
                      {rel.referenced_table}
                      {rel.on && ` on ${rel.on}`}
                    </Text>
                  </HStack>
                ))}
              </VStack>
            </Box>
          )}
        </Box>
      ))}
    </Box>
  );
};

export default SchemaRenderer; 