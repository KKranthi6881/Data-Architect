import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  HStack,
  Text,
  Badge,
  Button,
  Flex,
  Spinner,
  Icon,
  useColorModeValue,
  useToast,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Container,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  IconButton,
  Code,
  Divider,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Card,
  CardBody,
  Alert,
  AlertIcon,
  UnorderedList,
  ListItem
} from '@chakra-ui/react';
import { 
  IoCalendar, 
  IoChevronForward,
  IoTrashOutline,
  IoFilterOutline,
  IoDownloadOutline,
  IoDocumentTextOutline,
  IoChevronBack,
  IoArrowBack
} from 'react-icons/io5';
import { Link, useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Define API base URL directly in the component
const API_BASE_URL = 'http://localhost:8000';

const ChatHistoryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [conversationDetails, setConversationDetails] = useState(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();
  const { conversationId } = useParams();
  
  // For direct conversation fetching
  const [directConversation, setDirectConversation] = useState(null);
  const [directLoading, setDirectLoading] = useState(false);
  const [directError, setDirectError] = useState(null);

  // Theme colors
  const bgColor = useColorModeValue('white', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const primaryColor = useColorModeValue('gray.800', 'gray.100');
  const textSecondary = useColorModeValue('gray.600', 'gray.400');
  const messageBgUser = useColorModeValue('orange.50', 'orange.800');
  const messageBgAssistant = useColorModeValue('gray.50', 'gray.700');

  // Add new state for threads
  const [threads, setThreads] = useState([]);

  // Fetch conversations list
  useEffect(() => {
    console.log("ChatHistoryPage mounted, fetching conversations");
    fetchThreads();
  }, []);
  
  // Handle conversation loading when conversationId is in URL
  useEffect(() => {
    if (conversationId) {
      console.log(`Direct URL access to conversation: ${conversationId}`);
      loadDirectConversation(conversationId);
    } else {
      // Reset direct conversation view when no ID in URL
      setDirectConversation(null);
      setDirectLoading(false);
      setDirectError(null);
    }
  }, [conversationId]);

  // Update fetchThreads function
  const fetchThreads = async () => {
    try {
      setIsLoading(true);
      console.log("1. Starting to fetch threads...");
      const response = await fetch(`${API_BASE_URL}/api/thread-conversations`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch threads: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("2. API Response:", data);
      
      if (data.status === 'success' && Array.isArray(data.threads)) {
        console.log("3. Setting threads:", data.threads);
        setThreads(data.threads);
      } else {
        console.warn("4. Unexpected response format:", data);
        toast({
          title: 'Warning',
          description: 'Received unexpected data format from server',
          status: 'warning',
          duration: 5000,
        });
      }
    } catch (error) {
      console.error('5. Error fetching threads:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      console.log("6. Setting isLoading to false");
      setIsLoading(false);
    }
  };

  // Add this useEffect to monitor state changes
  useEffect(() => {
    console.log("Current state:", {
      isLoading,
      threadsLength: threads?.length,
      directConversation: !!directConversation
    });
  }, [isLoading, threads, directConversation]);

  // Main conversations list fetching
  const fetchConversations = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/conversations`);
      
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
          has_response: Boolean(conv.has_response),
          messages: conv.messages || []
        }));
        
        // Sort conversations by timestamp (newest first)
        const sortedConversations = validConversations.sort((a, b) => {
          return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        setConversations(sortedConversations);
        
        // If conversationId is in the URL, select that conversation
        if (conversationId) {
          const conversation = sortedConversations.find(c => c.id === conversationId);
          if (conversation) {
            setSelectedConversation(conversation);
          }
        }
      } else {
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
      setConversations([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to fetch a specific conversation directly
  const loadDirectConversation = async (id) => {
    try {
      setDirectLoading(true);
      setDirectError(null);
      
      console.log(`Fetching direct conversation: ${id}`);
      const response = await axios.get(`${API_BASE_URL}/api/conversation/${id}`);
      
      console.log('Direct conversation data:', response.data);
      setDirectConversation(response.data);
    } catch (error) {
      console.error('Error fetching direct conversation:', error);
      setDirectError(`Failed to load conversation: ${error.message}`);
      
      toast({
        title: 'Error',
        description: 'Unable to load the requested conversation',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setDirectLoading(false);
    }
  };
  
  // Other existing functions from your component
  const fetchConversationDetails = async (id) => {
    try {
      setIsLoadingDetail(true);
      console.log(`Fetching details for conversation: ${id}`);
      
      const response = await fetch(`${API_BASE_URL}/api/conversation/${id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch conversation details: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation details response:", data);
      
      if (data.status === 'success' && data.conversation) {
        const conversation = data.conversation;
        const messages = [];
        
        // Add user query message
        if (conversation.query) {
          messages.push({
            id: `${id}-user`,
            role: 'user',
            content: conversation.query,
            timestamp: conversation.timestamp || conversation.created_at
          });
        }
        
        // Add architect response message - use architect_response instead of output
        if (conversation.architect_response) {
          try {
            const architectResponse = typeof conversation.architect_response === 'string' 
              ? JSON.parse(conversation.architect_response)
              : conversation.architect_response;
              
            messages.push({
              id: `${id}-assistant`,
              role: 'assistant',
              type: 'architect',
              content: architectResponse.response || architectResponse,
              timestamp: conversation.timestamp || conversation.created_at,
              details: {
                sections: architectResponse.sections,
                schema_results: architectResponse.schema_results,
                code_results: architectResponse.code_results
              }
            });
          } catch (e) {
            console.error("Error parsing architect_response:", e);
            messages.push({
              id: `${id}-assistant`,
              role: 'assistant',
              content: "Error displaying response: " + e.message,
              timestamp: conversation.timestamp || conversation.created_at
            });
          }
        }
        
        setConversationDetails({
          id: id,
          thread_id: conversation.thread_id,
          messages: messages,
          metadata: conversation
        });
        
        const matchedConversation = conversations.find(c => c.id === id);
        if (matchedConversation) {
          setSelectedConversation(matchedConversation);
        } else {
          setSelectedConversation({
            id: id,
            timestamp: conversation.timestamp || conversation.created_at,
            preview: conversation.query || "No preview available",
            feedback_status: conversation.feedback_status || "pending"
          });
        }
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (error) {
      console.error('Error fetching conversation details:', error);
      toast({
        title: 'Error',
        description: `Failed to load conversation details: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const handleConversationSelect = (conversationId) => {
    console.log(`Selecting conversation: ${conversationId}`);
    
    // Update URL without full page reload
    navigate(`/history/${conversationId}`, { replace: false });
    
    // Set the selected conversation
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      setSelectedConversation(conversation);
      
      // Fetch full conversation details including messages
      fetchConversationDetails(conversationId);
    } else {
      toast({
        title: 'Error',
        description: 'Conversation not found',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const clearSelectedConversation = () => {
    setSelectedConversation(null);
    setConversationDetails(null);
    navigate('/history', { replace: true });
  };

  const handleDeleteConversation = async (conversationId, event) => {
    event.stopPropagation();
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete conversation');
      }
      
      setConversations(conversations.filter(conv => conv.id !== conversationId));
      
      toast({
        title: 'Conversation deleted',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error deleting conversation:', error);
      toast({
        title: 'Error deleting conversation',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown date';
    
    try {
      const date = new Date(dateStr);
      return date.toLocaleString();
    } catch (e) {
      console.error('Error formatting date:', e);
      return dateStr;
    }
  };

  const getTopicPreview = (conversation) => {
    if (conversation.preview) {
      return conversation.preview;
    }
    
    if (conversation.messages && conversation.messages.length > 0) {
      const firstUserMessage = conversation.messages.find(m => m.role === 'user');
      if (firstUserMessage && firstUserMessage.content) {
        return firstUserMessage.content.length > 80 
          ? firstUserMessage.content.substring(0, 80) + '...' 
          : firstUserMessage.content;
      }
    }
    
    if (conversation.title) {
      return conversation.title;
    }
    
    if (conversation.first_message) {
      return conversation.first_message.length > 80
        ? conversation.first_message.substring(0, 80) + '...'
        : conversation.first_message;
    }
    
    return 'Untitled Conversation';
  };
  
  const goToChat = (conversationId) => {
    console.log(`Navigating to chat with conversation: ${conversationId}`);
    navigate(`/chat/${conversationId}`);
  };

  // Format technical details for display
  const renderTechnicalDetails = (details) => {
    if (!details) return null;
    
    let parsedDetails = details;
    if (typeof details === 'string') {
      try {
        parsedDetails = JSON.parse(details);
      } catch (e) {
        console.error('Error parsing technical details:', e);
        return <Text color="red.500">Error parsing technical details</Text>;
      }
    }
    
    const parsedQuestion = parsedDetails.parsed_question || {};
    
    return (
      <Box mt={4}>
        <Heading size="md" mb={3}>Technical Details</Heading>
        
        {parsedQuestion.rephrased_question && (
          <Box mb={2}>
            <Text fontWeight="bold">Rephrased Question:</Text>
            <Text>{parsedQuestion.rephrased_question}</Text>
          </Box>
        )}
        
        {parsedQuestion.question_type && (
          <Box mb={2}>
            <Text fontWeight="bold">Question Type:</Text>
            <Badge 
              colorScheme={
                parsedQuestion.question_type === 'business' ? 'blue' : 
                parsedQuestion.question_type === 'technical' ? 'purple' : 
                'orange'
              }
            >
              {parsedQuestion.question_type.toUpperCase()}
            </Badge>
          </Box>
        )}
        
        {parsedQuestion.key_points && parsedQuestion.key_points.length > 0 && (
          <Box mb={2}>
            <Text fontWeight="bold">Key Points:</Text>
            <VStack align="start" pl={4}>
              {parsedQuestion.key_points.map((point, idx) => (
                <Text key={idx}>• {point}</Text>
              ))}
            </VStack>
          </Box>
        )}
      </Box>
    );
  };

  // Format code block
  const formatCodeBlock = (code, language = '') => {
    return (
      <Box borderRadius="md" overflow="hidden" my={2}>
        <SyntaxHighlighter
          language={language}
          style={atomDark}
          customStyle={{
            margin: 0,
            padding: '1rem',
            borderRadius: '0.375rem',
            fontSize: '0.9em',
          }}
        >
          {code}
        </SyntaxHighlighter>
      </Box>
    );
  };

  // If we're viewing a specific conversation by direct URL access
  if (conversationId && !selectedConversation) {
    // Loading state
    if (directLoading) {
      return (
        <Flex justify="center" align="center" minHeight="80vh">
          <Spinner size="xl" color="orange.500" />
        </Flex>
      );
    }
    
    // Error state
    if (directError) {
      return (
        <Box p={8}>
          <Button leftIcon={<IoArrowBack />} onClick={() => navigate('/history')} mb={4}>
            Back to Conversations
          </Button>
          <Alert status="error" borderRadius="md">
            <AlertIcon />
            {directError}
          </Alert>
        </Box>
      );
    }
    
    // Display direct conversation view
    if (directConversation) {
      return (
        <Box bg={bgColor} minH="calc(100vh - 64px)" py={8}>
          <Container maxW="container.xl">
            <HStack mb={6} spacing={4}>
              <Button 
                leftIcon={<IoArrowBack />} 
                onClick={() => navigate('/history')}
                variant="outline"
              >
                Back to Conversations
              </Button>
              <Heading size="lg" flex="1">Conversation Details</Heading>
              <Button 
                colorScheme="orange" 
                onClick={() => goToChat(directConversation.id)}
              >
                Continue in Chat
              </Button>
            </HStack>
            
            <Card borderWidth="1px" borderColor={borderColor} borderRadius="lg" mb={6}>
              <CardBody p={6}>
                <Text fontSize="sm" color="gray.500" mb={2}>
                  ID: {directConversation.id} • {formatDate(directConversation.timestamp)}
                </Text>
                
                <Divider my={4} />
                
                <Box mb={4}>
                  <Heading size="sm" mb={2}>Query:</Heading>
                  <Text p={3} bg="gray.50" borderRadius="md">
                    {directConversation.query}
                  </Text>
                </Box>
                
                <Box mb={4}>
                  <Heading size="sm" mb={2}>Response:</Heading>
                  <Box p={4} bg="gray.50" borderRadius="md">
                    {directConversation.architect_response ? (
                      <ReactMarkdown
                        components={{
                          code: ({node, inline, className, children, ...props}) => {
                            const match = /language-(\w+)/.exec(className || '');
                            return inline ? (
                              <Code colorScheme="orange" px={2} py={0.5} {...props}>
                                {children}
                              </Code>
                            ) : (
                              <Box 
                                bg="gray.800" 
                                borderRadius="md" 
                                p={4} 
                                my={4}
                                overflow="auto"
                              >
                                <SyntaxHighlighter
                                  language={match ? match[1] : ''}
                                  style={atomDark}
                                  customStyle={{
                                    margin: 0,
                                    background: 'transparent',
                                    fontSize: '0.9em',
                                  }}
                                >
                                  {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                              </Box>
                            );
                          },
                          p: ({children}) => (
                            <Text mb={4} lineHeight="tall">
                              {children}
                            </Text>
                          ),
                          ul: ({children}) => (
                            <UnorderedList spacing={2} pl={4} mb={4}>
                              {children}
                            </UnorderedList>
                          ),
                          li: ({children}) => (
                            <ListItem>
                              {children}
                            </ListItem>
                          ),
                          h1: ({children}) => (
                            <Heading as="h1" size="lg" mt={6} mb={4}>
                              {children}
                            </Heading>
                          ),
                          h2: ({children}) => (
                            <Heading as="h2" size="md" mt={5} mb={3}>
                              {children}
                            </Heading>
                          ),
                          h3: ({children}) => (
                            <Heading as="h3" size="sm" mt={4} mb={2}>
                              {children}
                            </Heading>
                          )
                        }}
                      >
                        {typeof directConversation.architect_response === 'string' 
                          ? directConversation.architect_response 
                          : directConversation.architect_response.response || JSON.stringify(directConversation.architect_response, null, 2)}
                      </ReactMarkdown>
                    ) : (
                      <Text color="gray.500">No response available</Text>
                    )}
                  </Box>
                </Box>
                
                {directConversation.technical_details && 
                  renderTechnicalDetails(directConversation.technical_details)
                }
                
                {directConversation.feedback && (
                  <Box mt={4}>
                    <Heading size="sm" mb={2}>Feedback</Heading>
                    <Badge 
                      colorScheme={
                        directConversation.feedback.status === 'approved' ? 'green' : 
                        directConversation.feedback.status === 'rejected' ? 'red' : 
                        'yellow'
                      }
                    >
                      {directConversation.feedback.status}
                    </Badge>
                    {directConversation.feedback.comments && (
                      <Text mt={1}>Comments: {directConversation.feedback.comments}</Text>
                    )}
                  </Box>
                )}
              </CardBody>
            </Card>
            
            {/* Debug section (hidden by default) */}
            <Box display="none">
              <Heading size="sm" mb={2}>Debug: Raw Data</Heading>
              <Code p={4} fontSize="xs" overflowX="auto" maxHeight="200px">
                {JSON.stringify(directConversation, null, 2)}
              </Code>
            </Box>
          </Container>
        </Box>
      );
    }
  }

  // Keep your existing render for when viewing the list or a selected conversation
  return (
    <Box bg={bgColor} minH="calc(100vh - 64px)" py={8}>
      {/* Debug element - remove this after fixing */}
      {selectedConversation && (
        <Box position="fixed" bottom="0" right="0" bg="black" color="white" p={4} zIndex={9999} maxW="300px" fontSize="xs">
          <Text fontWeight="bold">Debug Info:</Text>
          <Text>Selected: {selectedConversation?.id}</Text>
          <Text>Details: {conversationDetails ? 'Yes' : 'No'}</Text>
          <Text>Messages: {conversationDetails?.messages?.length || 0}</Text>
          <Button size="xs" onClick={() => {
            console.log("Debugging conversation details...");
            console.log("Conversation Details:", conversationDetails);
            console.log("Selected Conversation:", selectedConversation);
          }} mt={2}>
            Log Details
          </Button>
        </Box>
      )}

      <Container maxW="container.xl">
        {console.log("Rendering with:", { isLoading, threads })}
        
        {selectedConversation && conversationDetails ? (
          // Conversation detail view
          <Box>
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
                colorScheme="orange" 
                onClick={() => goToChat(selectedConversation.id)}
              >
                Continue in Chat
              </Button>
            </HStack>
            
            <Card mb={6} borderWidth="1px" borderColor={borderColor} borderRadius="lg">
              <CardBody>
                <HStack justify="space-between" mb={3}>
                  {selectedConversation.feedback_status && (
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
                  )}
                  <Text fontSize="sm" color="gray.500">
                    Started on {formatDate(selectedConversation.timestamp)}
                  </Text>
                </HStack>
                
                <Heading size="md" mb={2}>{getTopicPreview(selectedConversation)}</Heading>
                
                <Divider my={4} />
                
                {isLoadingDetail ? (
                  <Flex justify="center" align="center" height="300px">
                    <Spinner size="xl" color="orange.500" />
                  </Flex>
                ) : (
                  <Tabs variant="enclosed" colorScheme="orange">
                    <TabList>
                      <Tab>Conversation</Tab>
                      <Tab>Metadata</Tab>
                    </TabList>
                    
                    <TabPanels>
                      {/* Conversation Tab */}
                      <TabPanel>
                        <VStack align="stretch" spacing={4} p={2}>
                          {!conversationDetails ? (
                            <Text color="gray.500" textAlign="center" py={8}>
                              Loading conversation details...
                            </Text>
                          ) : conversationDetails.messages.length === 0 ? (
                            <Text color="gray.500" textAlign="center" py={8}>
                              No messages found for this conversation.
                            </Text>
                          ) : (
                            <Flex direction="column" width="100%">
                              {conversationDetails.messages.map((message, idx) => (
                                <Box
                                  key={message.id || `msg-${idx}`}
                                  p={4}
                                  mb={4}
                                  borderRadius="md"
                                  bg={message.role === 'user' ? messageBgUser : messageBgAssistant}
                                  borderWidth="1px"
                                  borderColor={borderColor}
                                >
                                  <HStack mb={2} justify="space-between">
                                    <Badge colorScheme={message.role === 'user' ? 'orange' : 'blue'}>
                                      {message.role === 'user' ? 'You' : 'Assistant'}
                                    </Badge>
                                    <Text fontSize="xs" color={textSecondary}>
                                      {formatDate(message.timestamp || '')}
                                    </Text>
                                  </HStack>
                                  
                                  <Text whiteSpace="pre-wrap">
                                    {message.content || "No content available"}
                                  </Text>
                                </Box>
                              ))}
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
                )}
              </CardBody>
            </Card>
          </Box>
        ) : (
          // Conversation list view (keep the current list view)
          <>
            <Flex justifyContent="space-between" alignItems="center" mb={6}>
              <Heading size="lg" color={primaryColor}>Conversation History</Heading>
              
              <HStack spacing={4}>
                <Button
                  leftIcon={<IoFilterOutline />}
                  variant="outline"
                  size="sm"
                >
                  Filter
                </Button>
                <Button
                  leftIcon={<IoDownloadOutline />}
                  variant="outline"
                  size="sm"
                >
                  Export
                </Button>
              </HStack>
            </Flex>
            
            {isLoading ? (
              <Flex justify="center" align="center" height="200px">
                <Spinner size="xl" color="orange.500" thickness="4px" />
              </Flex>
            ) : threads.length === 0 ? (
              <Box 
                p={10} 
                borderRadius="lg" 
                bg="gray.50" 
                textAlign="center"
                borderWidth="1px"
                borderColor={borderColor}
              >
                <Icon as={IoDocumentTextOutline} boxSize={12} color="gray.400" mb={4} />
                <Heading size="md" mb={2} color="gray.600">No conversations yet</Heading>
                <Text color="gray.500" mb={6}>Start a new chat to see your conversation history here.</Text>
                <Button 
                  as={Link} 
                  to="/chat" 
                  colorScheme="orange" 
                  leftIcon={<IoChevronForward />}
                >
                  Start a New Conversation
                </Button>
              </Box>
            ) : (
              <Accordion allowToggle width="100%">
                {threads.map((thread) => (
                  <AccordionItem 
                    key={thread.thread_id} 
                    border="1px solid" 
                    borderColor={borderColor}
                    borderRadius="md"
                    mb={3}
                  >
                    <AccordionButton py={4} px={5}>
                      <HStack flex="1" spacing={4} textAlign="left">
                        <Icon as={IoDocumentTextOutline} color="orange.500" boxSize={5} />
                        <Box>
                          <Text fontWeight="medium" fontSize="md">
                            {thread.initial_query}
                          </Text>
                          <Text fontSize="xs" color={textSecondary} mt={1}>
                            <Icon as={IoCalendar} boxSize={3} mr={1} />
                            {formatDate(thread.created_at)} • {thread.message_count} messages
                          </Text>
                        </Box>
                      </HStack>
                      <HStack spacing={3}>
                        <Badge colorScheme={
                          thread.has_architect_response ? 'green' : 'orange'
                        }>
                          {thread.has_architect_response ? 'Completed' : 'In Progress'}
                        </Badge>
                        <AccordionIcon />
                      </HStack>
                    </AccordionButton>
                    
                    <AccordionPanel pb={4}>
                      <VStack align="stretch" spacing={4}>
                        {thread.conversations.map((conv, idx) => (
                          <Box 
                            key={conv.id}
                            p={4}
                            bg={idx % 2 === 0 ? 'gray.50' : 'white'}
                            borderRadius="md"
                          >
                            <Text fontSize="sm" color="gray.500" mb={2}>
                              {formatDate(conv.created_at)}
                            </Text>
                            
                            {conv.query && (
                              <Box mb={3}>
                                <Text fontWeight="medium">Query:</Text>
                                <Text>{conv.query}</Text>
                              </Box>
                            )}

                            {conv.architect_response && (
                              <Box>
                                <Text fontWeight="medium">Architect Response:</Text>
                                <Box p={2} bg="white" borderRadius="md">
                                  <ReactMarkdown
                                    components={{
                                      code: ({node, inline, className, children, ...props}) => {
                                        const match = /language-(\w+)/.exec(className || '');
                                        return inline ? (
                                          <Code colorScheme="orange" px={2} py={0.5} {...props}>
                                            {children}
                                          </Code>
                                        ) : (
                                          <Box 
                                            bg="gray.800" 
                                            borderRadius="md" 
                                            p={4} 
                                            my={4}
                                            overflow="auto"
                                          >
                                            <SyntaxHighlighter
                                              language={match ? match[1] : ''}
                                              style={atomDark}
                                              customStyle={{
                                                margin: 0,
                                                background: 'transparent',
                                                fontSize: '0.9em',
                                              }}
                                            >
                                              {String(children).replace(/\n$/, '')}
                                            </SyntaxHighlighter>
                                          </Box>
                                        );
                                      }
                                    }}
                                  >
                                    {typeof conv.architect_response === 'string' 
                                      ? conv.architect_response 
                                      : conv.architect_response.response || JSON.stringify(conv.architect_response, null, 2)}
                                  </ReactMarkdown>
                                </Box>
                              </Box>
                            )}
                          </Box>
                        ))}
                      </VStack>
                    </AccordionPanel>
                  </AccordionItem>
                ))}
              </Accordion>
            )}
          </>
        )}
      </Container>
    </Box>
  );
};

export default ChatHistoryPage; 