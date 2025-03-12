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
  TabPanel
} from '@chakra-ui/react';
import { 
  IoCalendar, 
  IoChevronForward,
  IoTrashOutline,
  IoFilterOutline,
  IoDownloadOutline,
  IoDocumentTextOutline,
  IoChevronBack
} from 'react-icons/io5';
import { Link, useNavigate, useParams } from 'react-router-dom';

const ChatHistoryPage = () => {
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [conversationDetails, setConversationDetails] = useState(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();
  const { conversationId } = useParams();

  // Theme colors
  const bgColor = useColorModeValue('white', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const primaryColor = useColorModeValue('gray.800', 'gray.100');
  const textSecondary = useColorModeValue('gray.600', 'gray.400');
  const messageBgUser = useColorModeValue('orange.50', 'orange.800');
  const messageBgAssistant = useColorModeValue('gray.50', 'gray.700');

  useEffect(() => {
    console.log("ChatHistoryPage mounted, fetching conversations");
    fetchConversations();
  }, []);

  useEffect(() => {
    if (conversationId) {
      console.log("URL has conversation ID:", conversationId);
      
      // Always fetch the details when conversationId changes
      fetchConversationDetails(conversationId);
      
      // Select the conversation if we have it in our list
      const conversation = conversations.find(c => c.id === conversationId);
      if (conversation) {
        console.log("Found conversation in list, selecting:", conversation.id);
        setSelectedConversation(conversation);
      } else {
        console.log("Conversation not in list yet, creating placeholder");
        // Create a temporary placeholder
        setSelectedConversation({
          id: conversationId,
          timestamp: new Date().toISOString(),
          preview: "Loading conversation...",
          feedback_status: "pending"
        });
      }
    } else {
      // Clear selection if no conversationId in URL
      setSelectedConversation(null);
      setConversationDetails(null);
    }
  }, [conversationId, conversations]);

  const fetchConversations = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/conversations');
      
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

  // Fetch conversation details with thread messages
  const fetchConversationDetails = async (id) => {
    try {
      setIsLoadingDetail(true);
      console.log(`Fetching details for conversation: ${id}`);
      
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch conversation details: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation details response:", data);
      
      if (data.status === 'success' && data.conversation) {
        // Create a messages array with properly formatted messages
        const messages = [];
        const conversation = data.conversation;
        
        // Add user query message
        if (conversation.query) {
          messages.push({
            id: `${id}-user`,
            role: 'user',
            content: conversation.query,
            timestamp: conversation.timestamp || conversation.created_at
          });
        }
        
        // Add assistant response message - handle architect_response specifically
        if (conversation.architect_response) {
          // Parse the JSON string to get the actual response text
          try {
            const architectResponseObj = JSON.parse(conversation.architect_response);
            const responseText = architectResponseObj.response || "No response content";
            
            console.log("Extracted response text:", responseText.substring(0, 100) + "...");
            
            messages.push({
              id: `${id}-assistant`,
              role: 'assistant',
              content: responseText,  // Use the extracted text directly
              timestamp: conversation.timestamp || conversation.created_at
            });
          } catch (e) {
            console.error("Error parsing architect_response:", e);
            // Fall back to using the raw text
            messages.push({
              id: `${id}-assistant`,
              role: 'assistant',
              content: "Error displaying response: " + e.message,
              timestamp: conversation.timestamp || conversation.created_at
            });
          }
        } else if (conversation.response) {
          messages.push({
            id: `${id}-assistant`,
            role: 'assistant',
            content: conversation.response,
            timestamp: conversation.timestamp || conversation.created_at
          });
        }
        
        console.log("Processed messages:", messages);
        
        setConversationDetails({
          id: id,
          thread_id: conversation.thread_id,
          messages: messages,
          metadata: conversation
        });
        
        // Find and set the selected conversation
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
      const response = await fetch(`http://localhost:8000/api/conversations/${conversationId}`, {
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
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
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

  // Renders a single message in the detail view
  const renderMessage = (message) => {
    console.log("Rendering message:", message);
    
    // Extract content based on message structure
    let content = '';
    let role = '';
    
    if (message.role === 'user') {
      role = 'user';
      content = message.content || '';
    } else if (message.role === 'assistant') {
      role = 'assistant';
      content = message.content || '';
    } else {
      // If role is not explicitly defined, infer from content
      if (message.query) {
        role = 'user';
        content = message.query;
      } else if (message.response) {
        role = 'assistant';
        content = message.response;
      } else if (message.content) {
        // Default to user if unknown
        role = message.content.includes('?') ? 'user' : 'assistant';
        content = message.content;
      }
    }
    
    console.log(`Rendering message: role=${role}, content=${content.substring(0, 50)}...`);
    
    return (
      <Box 
        p={4} 
        mb={4} 
        borderRadius="md"
        bg={role === 'user' ? messageBgUser : messageBgAssistant}
        borderWidth="1px"
        borderColor={borderColor}
      >
        <HStack mb={2} justify="space-between">
          <Badge colorScheme={role === 'user' ? 'orange' : 'blue'}>
            {role === 'user' ? 'You' : 'Assistant'}
          </Badge>
          <Text fontSize="xs" color={textSecondary}>
            {formatDate(message.timestamp || message.created_at)}
          </Text>
        </HStack>
        
        <Text whiteSpace="pre-wrap">
          {content}
        </Text>
      </Box>
    );
  };

  // Debug function to help diagnose the issue
  const debugConversationDetails = () => {
    if (conversationDetails && conversationDetails.messages) {
      console.log("Messages to render:", conversationDetails.messages);
      
      // Check if messages have content
      conversationDetails.messages.forEach((msg, i) => {
        console.log(`Message ${i}:`, {
          role: msg.role,
          contentLength: msg.content ? msg.content.length : 0,
          content: msg.content ? msg.content.substring(0, 50) + '...' : 'MISSING',
        });
      });
    } else {
      console.log("No conversation details or messages available");
    }
  };

  return (
    <Box bg={bgColor} minH="calc(100vh - 64px)" py={8}>
      {/* Debug element - remove this after fixing */}
      {selectedConversation && (
        <Box position="fixed" bottom="0" right="0" bg="black" color="white" p={4} zIndex={9999} maxW="300px" fontSize="xs">
          <Text fontWeight="bold">Debug Info:</Text>
          <Text>Selected: {selectedConversation?.id}</Text>
          <Text>Details: {conversationDetails ? 'Yes' : 'No'}</Text>
          <Text>Messages: {conversationDetails?.messages?.length || 0}</Text>
          <Button size="xs" onClick={debugConversationDetails} mt={2}>
            Log Messages
          </Button>
        </Box>
      )}

      <Container maxW="container.xl">
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
            ) : conversations.length === 0 ? (
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
              <Accordion 
                allowToggle 
                defaultIndex={[]} 
                width="100%"
              >
                {conversations.map((conversation) => (
                  <AccordionItem 
                    key={conversation.id} 
                    border="1px solid" 
                    borderColor={borderColor}
                    borderRadius="md"
                    mb={3}
                    overflow="hidden"
                  >
                    <AccordionButton 
                      py={4} 
                      px={5}
                      _hover={{ bg: 'gray.50' }}
                      _expanded={{ bg: 'gray.50', fontWeight: 'medium' }}
                    >
                      <HStack flex="1" spacing={4} textAlign="left">
                        <Icon as={IoDocumentTextOutline} color="orange.500" boxSize={5} />
                        <Box>
                          <Text fontWeight="medium" fontSize="md">
                            {getTopicPreview(conversation)}
                          </Text>
                          <Text fontSize="xs" color={textSecondary} mt={1}>
                            <Icon as={IoCalendar} boxSize={3} mr={1} />
                            {formatDate(conversation.timestamp || conversation.created_at)} 
                            {conversation.messages && ` • ${conversation.messages.length} messages`}
                          </Text>
                        </Box>
                      </HStack>
                      <HStack spacing={3}>
                        <IconButton
                          icon={<IoTrashOutline />}
                          variant="ghost"
                          colorScheme="red"
                          size="sm"
                          aria-label="Delete conversation"
                          onClick={(e) => handleDeleteConversation(conversation.id, e)}
                        />
                        <Button
                          size="sm"
                          variant="outline"
                          colorScheme="orange"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleConversationSelect(conversation.id);
                          }}
                        >
                          View
                        </Button>
                        <AccordionIcon />
                      </HStack>
                    </AccordionButton>
                    
                    <AccordionPanel pb={4} bg="white">
                      <VStack align="stretch" spacing={4}>
                        <Box>
                          <Table variant="simple" size="sm">
                            <Thead bg="gray.50">
                              <Tr>
                                <Th>Details</Th>
                                <Th>Value</Th>
                              </Tr>
                            </Thead>
                            <Tbody>
                              <Tr>
                                <Td fontWeight="medium">Created</Td>
                                <Td>{formatDate(conversation.timestamp || conversation.created_at)}</Td>
                              </Tr>
                              {conversation.messages && (
                                <Tr>
                                  <Td fontWeight="medium">Messages</Td>
                                  <Td>{conversation.messages.length}</Td>
                                </Tr>
                              )}
                              <Tr>
                                <Td fontWeight="medium">ID</Td>
                                <Td>
                                  <Code fontSize="xs">{conversation.id}</Code>
                                </Td>
                              </Tr>
                            </Tbody>
                          </Table>
                        </Box>
                        
                        {conversation.messages && conversation.messages.length > 0 && (
                          <Box>
                            <Text fontWeight="medium" mb={2}>Messages Preview:</Text>
                            <VStack align="stretch" spacing={2} maxH="200px" overflowY="auto" px={2}>
                              {conversation.messages.slice(0, 3).map((message, idx) => (
                                <HStack 
                                  key={idx} 
                                  bg={message.role === 'user' ? 'orange.50' : 'gray.50'} 
                                  p={2} 
                                  borderRadius="md"
                                  borderLeftWidth="3px"
                                  borderLeftColor={message.role === 'user' ? 'orange.400' : 'gray.400'}
                                >
                                  <Badge colorScheme={message.role === 'user' ? 'orange' : 'gray'}>
                                    {message.role}
                                  </Badge>
                                  <Text fontSize="sm" noOfLines={2}>
                                    {message.content}
                                  </Text>
                                </HStack>
                              ))}
                              {conversation.messages.length > 3 && (
                                <Text fontSize="xs" color={textSecondary} textAlign="center">
                                  + {conversation.messages.length - 3} more messages
                                </Text>
                              )}
                            </VStack>
                          </Box>
                        )}
                        
                        <Divider />
                        
                        <HStack>
                          <Button
                            colorScheme="orange"
                            size="md"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleConversationSelect(conversation.id);
                            }}
                            flex="1"
                          >
                            View Full Conversation
                          </Button>
                          <Button
                            colorScheme="orange"
                            size="md"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              goToChat(conversation.id);
                            }}
                            flex="1"
                          >
                            Continue Chat
                          </Button>
                        </HStack>
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