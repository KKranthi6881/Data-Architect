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
  Icon,
  SimpleGrid,
  Input,
  InputGroup,
  InputLeftElement,
  useColorModeValue,
  useToast,
  Tab,
  Tabs,
  TabList,
  TabPanel,
  TabPanels,
  Avatar,
  Code,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Container
} from '@chakra-ui/react';
import { 
  IoSearch, 
  IoCalendar, 
  IoCheckmarkCircle, 
  IoTimeOutline,
  IoAlertCircleOutline,
  IoChevronForward,
  IoRefresh,
  IoChevronBack
} from 'react-icons/io5';
import { useNavigate, useParams } from 'react-router-dom';

const ChatHistoryPage = () => {
  console.log('ChatHistoryPage rendering...'); // Debug log

  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [conversationDetails, setConversationDetails] = useState(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const cardBg = useColorModeValue('white', 'gray.700');
  const hoverBg = useColorModeValue('gray.50', 'gray.600');
  const messageBgUser = useColorModeValue('blue.50', 'blue.800');
  const messageBgAssistant = useColorModeValue('white', 'gray.700');

  const bgColor = useColorModeValue('white', 'gray.900');
  const textColor = useColorModeValue('gray.900', 'white');

  // Fetch conversation history
  const fetchConversations = async () => {
    console.log("Fetching conversations from:", 'http://localhost:8000/api/conversations');
    
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/conversations');
      console.log("Response status:", response.status);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch conversations: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      console.log("API Response:", data); // Debug log
      
      if (data.status === 'success' && Array.isArray(data.conversations)) {
        // Ensure all required fields are present
        const validConversations = data.conversations.map(conv => ({
          id: conv.id || '',
          timestamp: conv.timestamp || conv.created_at || '',
          preview: conv.preview || 'No preview available',
          feedback_status: conv.feedback_status || 'pending',
          has_response: Boolean(conv.has_response)
        }));
        
        console.log("Processed conversations:", validConversations); // Debug log
        setConversations(validConversations);

        // If a conversationId is in the URL, select that conversation
        if (conversationId) {
          const conversation = validConversations.find(c => c.id === conversationId);
          if (conversation) {
            setSelectedConversation(conversation);
            fetchConversationDetails(conversationId);
          }
        }
      } else {
        console.warn("Unexpected API response format:", data);
        toast({
          title: 'Warning',
          description: 'Received unexpected response format from server',
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        setConversations([]);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      toast({
        title: 'Error',
        description: `Failed to load conversation history: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setConversations([]); // Clear conversations on error
    } finally {
      setIsLoading(false);
    }
  };

  // Load conversations on component mount
  useEffect(() => {
    console.log("ChatHistoryPage mounted, fetching conversations");
    fetchConversations();
    
    // Diagnostic log to check URLs
    console.log("Current pathname:", window.location.pathname);
    console.log("Current URL:", window.location.href);
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

  // Fetch details for a specific conversation
  const fetchConversationDetails = async (id) => {
    try {
      setIsLoadingDetail(true);
      console.log(`Fetching details for conversation: ${id}`);
      
      // First, fetch the basic conversation data
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch conversation details: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation details response:", data);
      
      if (data.status === 'success' && data.conversation) {
        // Get the thread ID from the conversation data
        const threadId = data.conversation.thread_id;
        
        if (threadId) {
          // Fetch all messages in this thread
          const threadResponse = await fetch(`http://localhost:8000/api/thread/${threadId}/conversations`);
          
          if (!threadResponse.ok) {
            throw new Error(`Failed to fetch thread conversations: ${threadResponse.status}`);
          }
          
          const threadData = await threadResponse.json();
          console.log("Thread data:", threadData);
          
          if (threadData.status === 'success') {
            // Format the conversation details
            const messages = [];
            
            // Process each conversation in the thread to create a message timeline
            threadData.conversations.forEach(conv => {
              // Add user message
              if (conv.query) {
                messages.push({
                  id: `${conv.id}-query`,
                  type: 'user',
                  content: conv.query,
                  timestamp: conv.timestamp
                });
              }
              
              // Add assistant response
              if (conv.response) {
                messages.push({
                  id: `${conv.id}-response`,
                  type: 'assistant',
                  content: conv.response,
                  timestamp: conv.timestamp,
                  details: {
                    conversation_id: conv.id,
                    feedback_status: conv.feedback_status || 'pending'
                  }
                });
              }
            });
            
            // Sort messages by timestamp if available
            messages.sort((a, b) => {
              if (!a.timestamp) return -1;
              if (!b.timestamp) return 1;
              return new Date(a.timestamp) - new Date(b.timestamp);
            });
            
            setConversationDetails({
              id: id,
              thread_id: threadId,
              messages: messages,
              metadata: data.conversation
            });
            
            // Update URL to include the conversation ID without navigating
            if (window.history.pushState) {
              const newUrl = `${window.location.pathname.split('/').slice(0, -1).join('/')}/${id}`;
              window.history.pushState({ path: newUrl }, '', newUrl);
            }
          }
        } else {
          // No thread ID, just display the single conversation
          const messages = [];
          
          if (data.conversation.query) {
            messages.push({
              id: `${id}-query`,
              type: 'user',
              content: data.conversation.query,
              timestamp: data.conversation.timestamp
            });
          }
          
          if (data.conversation.response) {
            messages.push({
              id: `${id}-response`,
              type: 'assistant',
              content: data.conversation.response,
              timestamp: data.conversation.timestamp,
              details: {
                conversation_id: id,
                feedback_status: data.conversation.feedback.status || 'pending'
              }
            });
          }
          
          setConversationDetails({
            id: id,
            messages: messages,
            metadata: data.conversation
          });
        }
      } else {
        toast({
          title: 'Error',
          description: 'Failed to load conversation details',
          status: 'error',
          duration: 3000,
        });
      }
    } catch (error) {
      console.error('Error fetching conversation details:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsLoadingDetail(false);
    }
  };

  // Handle conversation selection
  const handleConversationSelect = (id) => {
    console.log(`Selecting conversation: ${id}`);
    
    // Find the conversation in our list
    const conversation = conversations.find(conv => conv.id === id);
    if (conversation) {
      setSelectedConversation(conversation);
      fetchConversationDetails(id);
    } else {
      console.error(`Conversation with ID ${id} not found`);
    }
  };

  // Clear selected conversation
  const clearSelectedConversation = () => {
    setSelectedConversation(null);
    setConversationDetails(null);
    
    // Update URL to remove the conversation ID
    if (window.history.pushState) {
      const newUrl = window.location.pathname.split('/').slice(0, -1).join('/');
      window.history.pushState({ path: newUrl }, '', newUrl);
    }
  };

  // Go to chat
  const goToChat = (id) => {
    navigate(`/chat/${id}`);
  };

  // Start a new conversation
  const startNewConversation = () => {
    navigate('/chat');
  };

  // Filter conversations based on search query
  const filteredConversations = conversations.filter(conv => 
    conv.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  console.log("Rendering conversations:", filteredConversations.length, 
    filteredConversations.map(c => ({id: c.id, preview: c.preview.substring(0, 20)})));

  // Render a message in the conversation detail view
  const renderMessage = (message) => {
    return (
      <Box 
        key={message.id}
        bg={message.type === 'user' ? messageBgUser : messageBgAssistant}
        p={4}
        borderRadius="lg"
        boxShadow="sm"
        borderWidth="1px"
        borderColor={message.type === 'user' ? 'blue.100' : 'gray.200'}
        maxW="90%"
        alignSelf={message.type === 'user' ? 'flex-end' : 'flex-start'}
        mb={4}
      >
        <HStack mb={2} spacing={2}>
          {message.type === 'user' ? (
            <Avatar size="xs" bg="blue.500" />
          ) : (
            <Avatar size="xs" bg="green.500" />
          )}
          <Text 
            fontWeight="bold" 
            fontSize="sm"
            color={message.type === 'user' ? 'blue.600' : 'green.600'}
          >
            {message.type === 'user' ? 'You' : 'Assistant'}
          </Text>
          {message.timestamp && (
            <Text fontSize="xs" color="gray.500">
              {formatDate(message.timestamp)}
            </Text>
          )}
          {message.details?.feedback_status && (
            <Badge 
              colorScheme={
                message.details.feedback_status === 'approved' ? 'green' : 
                message.details.feedback_status === 'needs_improvement' ? 'orange' : 
                'blue'
              }
              fontSize="xs"
            >
              {message.details.feedback_status === 'approved' ? 'Approved' : 
               message.details.feedback_status === 'needs_improvement' ? 'Needs Review' : 
               'Pending'}
            </Badge>
          )}
        </HStack>
        
        <Text whiteSpace="pre-wrap">{message.content}</Text>
        
        {/* If the message has details, show them in an accordion */}
        {message.details && message.details.parsed_question && (
          <Accordion allowToggle mt={3}>
            <AccordionItem border="none">
              <AccordionButton px={0} _hover={{ bg: 'transparent' }}>
                <Text fontSize="xs" color="blue.500" fontWeight="medium">
                  Show Details
                </Text>
                <AccordionIcon ml={1} />
              </AccordionButton>
              <AccordionPanel pb={4} px={0}>
                <Code p={3} borderRadius="md" fontSize="xs" width="100%" overflowX="auto">
                  {JSON.stringify(message.details.parsed_question, null, 2)}
                </Code>
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        )}
      </Box>
    );
  };

  return (
    <Container maxW="container.xl">
      <VStack spacing={8} align="stretch" py={8}>
        <Heading color={textColor}>Chat History</Heading>
        <Text color="gray.600">
          View your previous conversations with the Data Architect Agent.
        </Text>
        {/* Main content */}
        {selectedConversation ? (
          // Conversation detail view
          <Box maxW="1200px" mx="auto" p={6}>
            <HStack mb={6} spacing={4}>
              <Button 
                leftIcon={<IoChevronBack />} 
                onClick={clearSelectedConversation}
                variant="outline"
              >
                Back to List
              </Button>
              <Heading size="lg" flex="1">Conversation Details</Heading>
              <Button 
                colorScheme="blue" 
                onClick={() => goToChat(selectedConversation.id)}
              >
                Continue in Chat
              </Button>
            </HStack>
            
            <Card mb={6}>
              <CardBody>
                <HStack justify="space-between" mb={3}>
                  <Badge 
                    colorScheme={
                      selectedConversation.feedback_status === 'approved' ? 'green' : 
                      selectedConversation.feedback_status === 'needs_improvement' ? 'orange' : 
                      'blue'
                    }
                    fontSize="sm"
                    p={2}
                  >
                    {selectedConversation.feedback_status === 'approved' ? 'Approved' : 
                     selectedConversation.feedback_status === 'needs_improvement' ? 'Needs Review' : 
                     'Pending'}
                  </Badge>
                  <Text fontSize="sm" color="gray.500">
                    Started on {formatDate(selectedConversation.timestamp)}
                  </Text>
                </HStack>
                
                <Heading size="md" mb={2}>{selectedConversation.preview}</Heading>
                
                <Divider my={4} />
                
                {isLoadingDetail ? (
                  <Flex justify="center" align="center" height="300px">
                    <Spinner size="xl" color="blue.500" />
                  </Flex>
                ) : conversationDetails ? (
                  <VStack align="stretch" spacing={0}>
                    <Tabs variant="enclosed" colorScheme="blue">
                      <TabList>
                        <Tab>Conversation</Tab>
                        <Tab>Metadata</Tab>
                      </TabList>
                      
                      <TabPanels>
                        {/* Conversation Tab */}
                        <TabPanel>
                          <VStack align="stretch" spacing={4} p={2}>
                            {conversationDetails.messages.length === 0 ? (
                              <Text color="gray.500" textAlign="center" py={8}>
                                No messages found for this conversation.
                              </Text>
                            ) : (
                              <Flex direction="column" width="100%">
                                {conversationDetails.messages.map(message => renderMessage(message))}
                              </Flex>
                            )}
                          </VStack>
                        </TabPanel>
                        
                        {/* Metadata Tab */}
                        <TabPanel>
                          <Box p={4} bg="gray.50" borderRadius="md">
                            <Heading size="sm" mb={3}>Conversation Metadata</Heading>
                            <Code p={4} borderRadius="md" width="100%" overflowX="auto">
                              {JSON.stringify(conversationDetails.metadata, null, 2)}
                            </Code>
                          </Box>
                        </TabPanel>
                      </TabPanels>
                    </Tabs>
                  </VStack>
                ) : (
                  <Text color="gray.500" textAlign="center" py={8}>
                    Failed to load conversation details.
                  </Text>
                )}
              </CardBody>
            </Card>
          </Box>
        ) : (
          // Conversation list view
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
              
              <HStack>
                <Button 
                  colorScheme="blue" 
                  onClick={startNewConversation}
                  mr={2}
                >
                  New Conversation
                </Button>
                <Button 
                  leftIcon={<IoRefresh />} 
                  onClick={fetchConversations}
                  isLoading={isLoading}
                  colorScheme="blue"
                  variant="outline"
                >
                  Refresh
                </Button>
              </HStack>
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
                  onClick={startNewConversation}
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
                    onClick={() => {
                      console.log(`Clicked on conversation: ${conversation.id}`);
                      handleConversationSelect(conversation.id);
                    }}
                  >
                    <CardBody>
                      <VStack align="stretch" spacing={3}>
                        <HStack justify="space-between">
                          <Badge 
                            colorScheme={
                              conversation.feedback_status === 'approved' ? 'green' : 
                              conversation.feedback_status === 'needs_improvement' ? 'orange' : 
                              'blue'
                            }
                            fontSize="xs"
                          >
                            {conversation.feedback_status === 'approved' ? 'Approved' : 
                             conversation.feedback_status === 'needs_improvement' ? 'Needs Review' : 
                             'Pending'}
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
                          {conversation.preview || "No preview available"}
                        </Text>
                        
                        {!conversation.has_response && (
                          <Badge colorScheme="yellow" fontSize="xs">
                            Awaiting Response
                          </Badge>
                        )}
                        
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
        )}
      </VStack>
    </Container>
  );
};

export default ChatHistoryPage; 