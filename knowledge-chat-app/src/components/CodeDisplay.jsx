import React from 'react';
import { Box, useColorModeValue, Text, Flex } from '@chakra-ui/react';
import { Prism } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

export const CodeDisplay = ({ code, language }) => {
  // Use dark styling for code blocks regardless of color mode
  const bgHeader = useColorModeValue('gray.100', 'gray.700');
  
  return (
    <Box 
      borderRadius="md" 
      overflow="hidden" 
      boxShadow="md"
      mb={4}
      borderWidth="1px"
      borderColor={useColorModeValue("gray.200", "gray.700")}
    >
      <Flex 
        bg={bgHeader} 
        p={2} 
        justifyContent="space-between" 
        alignItems="center"
        borderBottomWidth="1px"
        borderBottomColor={useColorModeValue("gray.200", "gray.600")}
      >
        <Text fontSize="xs" fontWeight="medium" color={useColorModeValue("gray.600", "gray.300")}>
          {language || 'code'}
        </Text>
      </Flex>
      
      <Prism
        language={language || 'sql'}
        style={atomDark}
        customStyle={{
          margin: 0,
          padding: '16px',
          fontSize: '0.9em',
          background: '#282c34',
          color: '#abb2bf',
          borderRadius: '0 0 0.375rem 0.375rem',
        }}
      >
        {code}
      </Prism>
    </Box>
  );
}; 