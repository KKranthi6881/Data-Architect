import React from 'react';
import {
  Box,
  Container,
  Flex,
  HStack,
  Heading,
  Text,
  Button,
  useColorModeValue,
  Icon
} from '@chakra-ui/react';
import { Link, useLocation } from 'react-router-dom';
import { 
  IoChatbubbles, 
  IoCloudUpload, 
  IoTime 
} from 'react-icons/io5';

const AppHeader = () => {
  const location = useLocation();
  const bgColor = useColorModeValue('white', 'gray.900');
  const borderColor = useColorModeValue('gray.100', 'gray.700');

  const isActive = (path) => location.pathname.startsWith(path);

  return (
    <Box bg={bgColor} borderBottom="1px" borderColor={borderColor}>
      <Container maxW="container.xl">
        <Flex justify="space-between" align="center" h="16">
          <Link to="/">
            <HStack spacing={2}>
              <Heading size="md">
                Data
                <Text as="span" color="orange.500">
                  NEURO
                </Text>
                <Text as="span" color="gray.500">
                  .AI
                </Text>
              </Heading>
            </HStack>
          </Link>

          <HStack spacing={4}>
            <Button
              as={Link}
              to="/chat"
              leftIcon={<IoChatbubbles />}
              variant={isActive('/chat') ? 'solid' : 'ghost'}
              colorScheme="orange"
            >
              Chat
            </Button>
            <Button
              as={Link}
              to="/upload"
              leftIcon={<IoCloudUpload />}
              variant={isActive('/upload') ? 'solid' : 'ghost'}
              colorScheme="orange"
            >
              Upload
            </Button>
            <Button
              as={Link}
              to="/history"
              leftIcon={<IoTime />}
              variant={isActive('/history') ? 'solid' : 'ghost'}
              colorScheme="orange"
            >
              History
            </Button>
          </HStack>
        </Flex>
      </Container>
    </Box>
  );
};

export default AppHeader; 