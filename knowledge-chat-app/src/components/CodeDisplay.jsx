import React from 'react';
import { Box, useColorModeValue } from '@chakra-ui/react';
import { Prism } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

export const CodeDisplay = ({ code, language }) => {
  const bgColor = useColorModeValue('gray.50', 'gray.700');
  
  return (
    <Box borderRadius="md" overflow="hidden" bg={bgColor}>
      <Prism
        language={language || 'sql'}
        style={tomorrow}
        customStyle={{
          margin: 0,
          borderRadius: '0.375rem',
        }}
      >
        {code}
      </Prism>
    </Box>
  );
}; 