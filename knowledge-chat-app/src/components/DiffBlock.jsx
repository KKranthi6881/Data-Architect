import React from 'react';
import { Box, Text, Flex, IconButton, HStack, Badge, useClipboard } from '@chakra-ui/react';
import { IoCopy, IoCheckmarkDone } from 'react-icons/io5';
import Prism from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * DiffBlock component to display code differences with syntax highlighting.
 * This component accepts a diff string and renders it with proper formatting
 * for additions (green) and deletions (red).
 */
const DiffBlock = ({ diff, language = 'sql' }) => {
  const { hasCopied, onCopy } = useClipboard(diff);
  
  // Process the diff lines
  const diffLines = diff.split('\n').map((line, index) => {
    let backgroundColor = 'transparent';
    let color = '#abb2bf';
    
    if (line.startsWith('+') && !line.startsWith('+++')) {
      backgroundColor = 'rgba(0, 170, 0, 0.2)';
      color = '#b5f3b5';
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      backgroundColor = 'rgba(170, 0, 0, 0.2)';
      color = '#f8bdbd';
    } else if (line.startsWith('@@ ')) {
      backgroundColor = 'rgba(0, 92, 197, 0.2)';
      color = '#79b8ff';
    }
    
    return (
      <Text
        key={index}
        py={0.5}
        px={2}
        fontFamily="monospace"
        backgroundColor={backgroundColor}
        color={color}
        whiteSpace="pre"
      >
        {line}
      </Text>
    );
  });

  return (
    <Box
      mt={4}
      mb={4}
      borderRadius="md"
      overflow="hidden"
      boxShadow="md"
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
        borderBottom="1px solid"
        borderColor="gray.700"
      >
        <Badge colorScheme="purple" variant="solid">diff</Badge>
        <IconButton
          icon={hasCopied ? <IoCheckmarkDone /> : <IoCopy />}
          size="sm"
          variant="ghost"
          color="gray.300"
          _hover={{ bg: 'gray.700' }}
          colorScheme={hasCopied ? "green" : "gray"}
          onClick={onCopy}
          aria-label="Copy diff"
          title="Copy to clipboard"
        />
      </HStack>
      <Box
        bg="#282c34"
        maxHeight="500px"
        overflowY="auto"
        fontSize="sm"
      >
        {diffLines}
      </Box>
    </Box>
  );
};

export default DiffBlock; 