import React from 'react';
import { 
  Box, 
  Flex, 
  Heading, 
  Button, 
  HStack,
  useColorModeValue,
  Icon
} from '@chakra-ui/react';
import { Link, useLocation } from 'react-router-dom';
import { IoChat, IoCloudUpload, IoTime } from 'react-icons/io5';

const Header = () => {
  const location = useLocation();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Check which tab is active
  const isActive = (path) => {
    if (path === '/chat') {
      return location.pathname === '/' || location.pathname === '/chat' || location.pathname.startsWith('/chat/');
    }
    return location.pathname === path;
  };

  return (
    <Box 
      as="header" 
      py={4} 
      px={6} 
      bg={bgColor} 
      boxShadow="sm"
      borderBottomWidth="1px"
      borderColor={borderColor}
    >
      <Flex 
        maxW="1200px" 
        mx="auto" 
        justify="space-between" 
        align="center"
      >
        <Heading 
          size="md" 
          color="blue.600"
          fontWeight="bold"
        >
          Knowledge Chat
        </Heading>
        
        <HStack spacing={4}>
          <Button
            as={Link}
            to="/chat"
            variant={isActive('/chat') ? "solid" : "ghost"}
            colorScheme="blue"
            leftIcon={<Icon as={IoChat} />}
            size="md"
          >
            Chat
          </Button>
          
          <Button
            as={Link}
            to="/history"
            variant={isActive('/history') ? "solid" : "ghost"}
            colorScheme="blue"
            leftIcon={<Icon as={IoTime} />}
            size="md"
          >
            History
          </Button>
          
          <Button
            as={Link}
            to="/upload"
            variant={isActive('/upload') ? "solid" : "ghost"}
            colorScheme="blue"
            leftIcon={<Icon as={IoCloudUpload} />}
            size="md"
          >
            Upload
          </Button>
        </HStack>
      </Flex>
    </Box>
  );
};

export default Header; 