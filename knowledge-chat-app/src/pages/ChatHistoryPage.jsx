import React, { useState, useEffect, useRef } from 'react';
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
  ListItem,
  CardHeader,
  SimpleGrid
} from '@chakra-ui/react';
import { 
  IoCalendar, 
  IoChevronForward,
  IoTrashOutline,
  IoFilterOutline,
  IoDownloadOutline,
  IoDocumentTextOutline,
  IoChevronBack,
  IoArrowBack,
  IoTrash,
  IoRefresh
} from 'react-icons/io5';
import { Link, useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import remarkGfm from 'remark-gfm';

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
  const { conversationId, threadId } = useParams();
  
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

  // State for selected thread
  const [selectedThread, setSelectedThread] = useState(null);
  const [threadConversations, setThreadConversations] = useState([]);
  const [isLoadingThread, setIsLoadingThread] = useState(false);
  
  // Colors
  const cardBg = useColorModeValue('white', 'gray.800');
  const hoverBg = useColorModeValue('gray.50', 'gray.700');

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

  // Load thread if threadId is provided in URL
  useEffect(() => {
    if (threadId) {
      handleThreadSelect(threadId);
    }
  }, [threadId]);

  // Update fetchThreads function
  const fetchThreads = async () => {
    try {
      setIsLoading(true);
      console.log("Starting to fetch threads...");
      
      const response = await fetch(`${API_BASE_URL}/api/thread-conversations`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch threads: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("API Response:", data);
      
      // Check if threads array exists in the response
      if (Array.isArray(data.threads)) {
        setThreads(data.threads);
      } else {
        console.warn("Unexpected response format:", data);
        setThreads([]);
        toast({
          title: 'Warning',
          description: 'Received unexpected data format from server',
          status: 'warning',
          duration: 5000,
        });
      }
    } catch (error) {
      console.error('Error fetching threads:', error);
      setThreads([]);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
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

  // Add this helper function for date formatting
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    
    try {
      const date = new Date(dateString);
      
      // Check if date is valid
      if (isNaN(date.getTime())) {
        return 'Invalid date';
      }
      
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      console.error('Error formatting date:', error);
      return 'Date error';
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
        <Prism
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
        </Prism>
      </Box>
    );
  };

  // Fetch conversations for a specific thread
  const fetchThreadConversations = async (threadId) => {
    try {
      setIsLoadingThread(true);
      console.log(`Fetching conversations for thread: ${threadId}`);
      
      // First try the thread-conversations endpoint
      let response = await fetch(`${API_BASE_URL}/api/thread-conversations/${threadId}`);
      
      // If that fails, try to get the individual conversation
      if (!response.ok) {
        console.log(`Thread endpoint failed, trying individual conversation: ${threadId}`);
        response = await fetch(`${API_BASE_URL}/api/conversation/${threadId}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch conversation: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("Individual conversation response:", data);
        
        // Convert the individual conversation to an array with proper field mapping
        setThreadConversations([{
          conversation_id: data.id || threadId,
          thread_id: threadId,
          question: data.query || "No question available",
          answer: data.response || "No response available",
          metadata: data.technical_details || {},
          timestamp: data.timestamp || new Date().toISOString()
        }]);
        
        return;
      }
      
      const data = await response.json();
      console.log("Thread conversations response:", data);
      
      if (Array.isArray(data.conversations)) {
        // Make sure we have valid data in each conversation
        const validatedConversations = data.conversations.map(conv => ({
          ...conv,
          question: conv.question || "No question available",
          answer: conv.answer || "No response available"
        }));
        
        setThreadConversations(validatedConversations);
      } else {
        console.warn("Unexpected response format:", data);
        setThreadConversations([]);
        toast({
          title: 'Warning',
          description: 'Received unexpected data format from server',
          status: 'warning',
          duration: 5000,
        });
      }
    } catch (error) {
      console.error('Error fetching thread conversations:', error);
      setThreadConversations([]);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsLoadingThread(false);
    }
  };
  
  // Handle thread selection
  const handleThreadSelect = (threadId) => {
    console.log(`Selecting thread: ${threadId}`);
    
    // Update URL without full page reload
    navigate(`/history/${threadId}`, { replace: false });
    
    // Find the thread in our list
    const thread = threads.find(t => t.thread_id === threadId);
    if (thread) {
      setSelectedThread(thread);
      
      // Fetch conversations for this thread
      fetchThreadConversations(threadId);
    } else {
      // If thread not in our list, fetch it directly
      fetchThreadConversations(threadId);
      
      // Create a placeholder thread
      setSelectedThread({
        thread_id: threadId,
        latest_question: "Loading...",
        conversation_count: 0
      });
    }
  };
  
  // Clear selected thread
  const clearSelectedThread = () => {
    setSelectedThread(null);
    setThreadConversations([]);
    navigate('/history', { replace: true });
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
                                <Prism
                                  language={match ? match[1] : ''}
                                  style={atomDark}
                                  customStyle={{
                                    margin: 0,
                                    background: 'transparent',
                                    fontSize: '0.9em',
                                  }}
                                >
                                  {String(children).replace(/\n$/, '')}
                                </Prism>
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
        
        {selectedThread ? (
          // Thread detail view
          <Box>
            <Button 
              leftIcon={<IoArrowBack />} 
              variant="outline" 
              mb={4}
              onClick={clearSelectedThread}
            >
              Back to All Threads
            </Button>
            
            <Card borderWidth="1px" borderColor={borderColor} mb={4}>
              <CardHeader>
                <HStack justify="space-between">
                  <Heading size="md">
                    Thread: {selectedThread.latest_question || "Untitled Thread"}
                  </Heading>
                  <Badge colorScheme="blue">
                    {selectedThread.conversation_count || threadConversations.length} messages
                  </Badge>
                </HStack>
                <Text fontSize="sm" color="gray.500" mt={2}>
                  Last updated: {formatDate(selectedThread.latest_timestamp)}
                </Text>
              </CardHeader>
              
              <CardBody>
                {isLoadingThread ? (
                  <Flex justify="center" py={8}>
                    <Spinner size="xl" />
                  </Flex>
                ) : threadConversations.length === 0 ? (
                  <Text color="gray.500" textAlign="center" py={4}>
                    No conversations found in this thread
                  </Text>
                ) : (
                  <>
                    <Button 
                      size="xs" 
                      colorScheme="blue" 
                      variant="outline" 
                      mb={4}
                      onClick={() => {
                        console.log("Thread conversations:", threadConversations);
                      }}
                    >
                      Debug Conversations
                    </Button>
                    
                    <VStack spacing={6} align="stretch">
                      {threadConversations.map((conv, index) => (
                        <Box key={conv.conversation_id} borderWidth="1px" borderRadius="md" p={4}>
                          <Text fontWeight="bold" mb={2}>
                            Question:
                          </Text>
                          <Box bg="blue.50" p={3} borderRadius="md" mb={4}>
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code({node, inline, className, children, ...props}) {
                                  const match = /language-(\w+)/.exec(className || '')
                                  return !inline && match ? (
                                    <Prism
                                      style={atomDark}
                                      language={match[1]}
                                      PreTag="div"
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, '')}
                                    </Prism>
                                  ) : (
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  )
                                }
                              }}
                            >
                              {conv.question}
                            </ReactMarkdown>
                          </Box>
                          
                          <Text fontWeight="bold" mb={2}>
                            Answer:
                          </Text>
                          <Box bg="gray.50" p={3} borderRadius="md">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                code({node, inline, className, children, ...props}) {
                                  const match = /language-(\w+)/.exec(className || '')
                                  return !inline && match ? (
                                    <Prism
                                      style={atomDark}
                                      language={match[1]}
                                      PreTag="div"
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, '')}
                                    </Prism>
                                  ) : (
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  )
                                }
                              }}
                            >
                              {conv.answer}
                            </ReactMarkdown>
                          </Box>
                          
                          <Divider my={3} />
                          
                          <HStack justify="space-between">
                            <Text fontSize="sm" color="gray.500">
                              {formatDate(conv.timestamp)}
                            </Text>
                            <Badge colorScheme="purple">
                              {conv.metadata?.question_type || "General"}
                            </Badge>
                          </HStack>
                        </Box>
                      ))}
                    </VStack>
                  </>
                )}
              </CardBody>
            </Card>
          </Box>
        ) : (
          // Threads list view
          <Box>
            <HStack justify="space-between" mb={4}>
              <Heading size="md">All Conversation Threads</Heading>
              <Button 
                leftIcon={<IoRefresh />}
                onClick={fetchThreads}
                isLoading={isLoading}
              >
                Refresh
              </Button>
            </HStack>
            
            {isLoading ? (
              <Flex justify="center" py={8}>
                <Spinner size="xl" />
              </Flex>
            ) : threads.length === 0 ? (
              <Text color="gray.500" textAlign="center" py={8}>
                No conversation threads found
              </Text>
            ) : (
              <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4}>
                {threads.map(thread => (
                  <Card 
                    key={thread.thread_id}
                    borderWidth="1px"
                    borderColor={borderColor}
                    bg={cardBg}
                    _hover={{ bg: hoverBg, cursor: 'pointer' }}
                    onClick={() => handleThreadSelect(thread.thread_id)}
                  >
                    <CardBody>
                      <VStack align="start" spacing={2}>
                        <Heading size="sm" noOfLines={2}>
                          {thread.latest_question || "Untitled Thread"}
                        </Heading>
                        
                        <HStack justify="space-between" w="100%">
                          <Badge colorScheme="blue">
                            {thread.conversation_count} messages
                          </Badge>
                          <Text fontSize="xs" color="gray.500">
                            {formatDate(thread.latest_timestamp)}
                          </Text>
                        </HStack>
                        
                        <HStack w="100%" justify="flex-end">
                          <IconButton
                            icon={<IoChevronForward />}
                            variant="ghost"
                            size="sm"
                            aria-label="View thread"
                          />
                        </HStack>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            )}
          </Box>
        )}
      </Container>
    </Box>
  );
};

export default ChatHistoryPage; 