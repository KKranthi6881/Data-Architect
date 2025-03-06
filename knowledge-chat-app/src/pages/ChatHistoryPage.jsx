import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  HStack,
  Text,
  Badge,
  Card,
  CardBody,
  Divider,
  Button,
  Flex,
  Spinner,
  useToast,
  Icon,
  SimpleGrid,
  Input,
  InputGroup,
  InputLeftElement,
  useColorModeValue
} from '@chakra-ui/react';
import { 
  IoSearch, 
  IoCalendar, 
  IoCheckmarkCircle, 
  IoTimeOutline,
  IoAlertCircleOutline,
  IoChevronForward,
  IoRefresh
} from 'react-icons/io5';
import { useNavigate } from 'react-router-dom';

const ChatHistoryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const toast = useToast();
  const navigate = useNavigate();
  const cardBg = useColorModeValue('white', 'gray.700');
  const hoverBg = useColorModeValue('gray.50', 'gray.600');

  // Fetch conversation history
  const fetchConversations = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/conversations');
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }
      const data = await response.json();
      console.log("Conversations data:", data);
      
      if (data.status === 'success') {
        setConversations(data.conversations);
      } else {
        throw new Error(data.message || 'Failed to load conversations');
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Load conversations on component mount
  useEffect(() => {
    fetchConversations();
  }, []);

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    
    try {
      const date = new Date(dateString);
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    } catch (e) {
      return dateString;
    }
  };

  // Handle conversation selection
  const handleConversationSelect = (id) => {
    navigate(`/chat/${id}`);
  };

  // Filter conversations based on search query
  const filteredConversations = conversations.filter(conv => 
    conv.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Box maxW="1200px" mx="auto" p={6}>
      <Heading size="lg" mb={6}>Conversation History</Heading>
      
      {/* Search and refresh controls */}
      <Flex mb={6} justifyContent="space-between" alignItems="center">
        <InputGroup maxW="400px">
          <InputLeftElement pointerEvents="none">
            <Icon as={IoSearch} color="gray.400" />
          </InputLeftElement>
          <Input 
            placeholder="Search conversations..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </InputGroup>
        
        <Button 
          leftIcon={<IoRefresh />} 
          onClick={fetchConversations}
          isLoading={isLoading}
          colorScheme="blue"
          variant="outline"
        >
          Refresh
        </Button>
      </Flex>
      
      {/* Conversations list */}
      {isLoading ? (
        <Flex justify="center" align="center" height="300px">
          <Spinner size="xl" color="blue.500" />
        </Flex>
      ) : filteredConversations.length === 0 ? (
        <Box textAlign="center" p={10} bg="gray.50" borderRadius="md">
          <Text fontSize="lg" color="gray.500">
            {searchQuery ? 'No conversations match your search' : 'No conversations found'}
          </Text>
          <Button 
            mt={4} 
            colorScheme="blue" 
            onClick={() => navigate('/chat')}
          >
            Start a new conversation
          </Button>
        </Box>
      ) : (
        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {filteredConversations.map((conversation) => (
            <Card 
              key={conversation.id} 
              bg={cardBg}
              borderRadius="lg"
              boxShadow="md"
              cursor="pointer"
              transition="all 0.2s"
              _hover={{ 
                transform: 'translateY(-4px)', 
                boxShadow: 'lg',
                bg: hoverBg
              }}
              onClick={() => handleConversationSelect(conversation.id)}
            >
              <CardBody>
                <VStack align="stretch" spacing={3}>
                  <HStack justify="space-between">
                    <Badge 
                      colorScheme={
                        conversation.feedback_status === 'approved' ? 'green' : 
                        conversation.feedback_status === 'needs_improvement' ? 'orange' : 'blue'
                      }
                      fontSize="xs"
                    >
                      {
                        conversation.feedback_status === 'approved' ? 'Approved' : 
                        conversation.feedback_status === 'needs_improvement' ? 'Needs Review' : 'Pending'
                      }
                    </Badge>
                    <HStack spacing={1}>
                      <Icon as={IoTimeOutline} color="gray.500" boxSize={3} />
                      <Text fontSize="xs" color="gray.500">
                        {formatDate(conversation.timestamp)}
                      </Text>
                    </HStack>
                  </HStack>
                  
                  <Text 
                    fontWeight="medium" 
                    fontSize="md" 
                    noOfLines={2}
                  >
                    {conversation.preview}
                  </Text>
                  
                  <Divider />
                  
                  <HStack justify="flex-end">
                    <Text fontSize="sm" color="blue.500">
                      View Conversation
                    </Text>
                    <Icon as={IoChevronForward} color="blue.500" />
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>
      )}
    </Box>
  );
};

export default ChatHistoryPage; 