import React from 'react';
import {
  Box,
  Flex,
  Heading,
  Button,
  HStack,
  useColorModeValue,
  Spacer,
  Icon
} from '@chakra-ui/react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { IoCheckmarkCircle, IoChatbubbles, IoTime, IoHome } from 'react-icons/io5';

const AppHeader = () => {
  const location = useLocation();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Check current route to highlight active link
  const isActive = (path) => {
    return location.pathname.startsWith(path);
  };
  
  return (
    <Box 
      as="header" 
      bg={bgColor} 
      borderBottomWidth="1px" 
      borderColor={borderColor}
      position="sticky"
      top="0"
      zIndex="10"
      boxShadow="sm"
      width="100%"
    >
      <Flex 
        maxW="1400px" 
        mx="auto" 
        px={4} 
        py={3} 
        align="center"
      >
        <Heading 
          size="md" 
          color="blue.600"
          display="flex"
          alignItems="center"
        >
          <Icon as={IoCheckmarkCircle} mr={2} />
          Knowledge Chat
        </Heading>
        
        <Spacer />
        
        <HStack spacing={2}>
          <Button
            as={RouterLink}
            to="/"
            colorScheme={isActive('/') && !isActive('/chat') && !isActive('/history') ? 'blue' : 'gray'}
            variant={isActive('/') && !isActive('/chat') && !isActive('/history') ? 'solid' : 'ghost'}
            leftIcon={<IoHome />}
            size="md"
          >
            Home
          </Button>
          
          <Button
            as={RouterLink}
            to="/chat"
            colorScheme={isActive('/chat') ? 'blue' : 'gray'}
            variant={isActive('/chat') ? 'solid' : 'ghost'}
            leftIcon={<IoChatbubbles />}
            size="md"
          >
            Chat
          </Button>
          
          <Button
            as={RouterLink}
            to="/history"
            colorScheme={isActive('/history') ? 'blue' : 'gray'}
            variant={isActive('/history') ? 'solid' : 'ghost'}
            leftIcon={<IoTime />}
            size="md"
          >
            History
          </Button>
        </HStack>
      </Flex>
    </Box>
  );
};

export default AppHeader; 