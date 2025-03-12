import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Box, 
  VStack,
  Text,
  Heading,
  Card,
  CardBody,
  Badge,
  Flex,
  Button,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
  useToast
} from '@chakra-ui/react';
import { IoAdd, IoTrash } from 'react-icons/io5';
import { fetchRecentConversations, fetchConversationsByThread, clearConversation } from '../api/chatApi';

const ConversationHistory = ({ onSelectConversation, onNewChat }) => {
  const [threads, setThreads] = useState([]);
  const [expandedThreads, setExpandedThreads] = useState({});
  const [threadConversations, setThreadConversations] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const toast = useToast();

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    try {
      setLoading(true);
      const response = await fetchRecentConversations();
      
      if (response.status === 'success') {
        // The backend now returns threads instead of individual conversations
        setThreads(response.conversations);
        console.log("Loaded threads:", response.conversations);
      }
    } catch (err) {
      console.error('Error loading conversations:', err);
      setError(err.message);
      toast({
        title: 'Error',
        description: 'Failed to load conversation history',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleThreadExpansion = async (threadId) => {
    // Toggle expansion state
    setExpandedThreads(prev => ({
      ...prev,
      [threadId]: !prev[threadId]
    }));
    
    // If expanding and we don't have conversations for this thread yet, fetch them
    if (!expandedThreads[threadId] && !threadConversations[threadId]) {
      try {
        const response = await fetchConversationsByThread(threadId);
        console.log('Thread conversations response:', response); // Debug log
        
        if (response.status === 'success') {
          // Transform the conversations to include architect response
          const enhancedConversations = response.conversations.map(conv => {
            // Parse technical details to get architect response
            let architectResponse = null;
            try {
              const technicalDetails = JSON.parse(conv.technical_details || '{}');
              if (technicalDetails.architect_response) {
                architectResponse = technicalDetails.architect_response;
              }
            } catch (err) {
              console.error('Error parsing technical details:', err);
            }

            return {
              ...conv,
              architect_response: architectResponse
            };
          });

          setThreadConversations(prev => ({
            ...prev,
            [threadId]: enhancedConversations
          }));
          console.log(`Enhanced conversations for thread ${threadId}:`, enhancedConversations);
        }
      } catch (err) {
        console.error(`Error loading conversations for thread ${threadId}:`, err);
        toast({
          title: 'Error',
          description: 'Failed to load thread conversations',
          status: 'error',
          duration: 3000,
        });
      }
    }
  };

  const handleClearConversation = async (conversationId, e) => {
    e.stopPropagation(); // Prevent triggering the parent click event
    
    try {
      await clearConversation(conversationId);
      toast({
        title: 'Conversation cleared',
        status: 'success',
        duration: 2000,
      });
      
      // Refresh the conversation list
      loadConversations();
    } catch (err) {
      console.error('Error clearing conversation:', err);
      toast({
        title: 'Error',
        description: 'Failed to clear conversation',
        status: 'error',
        duration: 3000,
      });
    }
  };

  const renderThreadConversations = (conversations) => {
    return (
      <VStack spacing={2} align="stretch">
        {conversations.map((conv, index) => {
          // Parse technical details if needed
          let architectResponse = conv.architect_response;
          if (!architectResponse && conv.technical_details) {
            try {
              const technicalDetails = JSON.parse(conv.technical_details);
              architectResponse = technicalDetails.architect_response;
            } catch (err) {
              console.error('Error parsing technical details:', err);
            }
          }

          return (
            <Box 
              key={conv.id}
              borderWidth="1px"
              borderRadius="md"
              p={3}
              bg={architectResponse ? "purple.50" : "white"}
            >
              {/* User Query */}
              <Flex direction="column" mb={2}>
                <Text fontSize="sm" fontWeight="medium">
                  <Badge colorScheme="blue" mr={2}>User</Badge>
                  {conv.query}
                </Text>
                <Text fontSize="xs" color="gray.500">
                  {new Date(conv.timestamp).toLocaleString()}
                </Text>
              </Flex>

              {/* Assistant Response */}
              {conv.response && (
                <Flex direction="column" mb={2} pl={4}>
                  <Text fontSize="sm">
                    <Badge colorScheme="green" mr={2}>Assistant</Badge>
                    {conv.response}
                  </Text>
                </Flex>
              )}

              {/* Data Architect Response */}
              {architectResponse && (
                <Flex direction="column" pl={4} mt={2}>
                  <Text fontSize="sm">
                    <Badge colorScheme="purple" mr={2}>Data Architect</Badge>
                    {architectResponse.response}
                  </Text>
                  {architectResponse.sections && (
                    <Accordion allowToggle size="sm" mt={2}>
                      <AccordionItem>
                        <AccordionButton>
                          <Box flex="1" textAlign="left">
                            View Details
                          </Box>
                          <AccordionIcon />
                        </AccordionButton>
                        <AccordionPanel>
                          {Object.entries(architectResponse.sections).map(([key, value]) => (
                            <Box key={key} mb={2}>
                              <Text fontSize="sm" fontWeight="bold" color="purple.600">
                                {key.replace(/_/g, ' ').toUpperCase()}
                              </Text>
                              <Text fontSize="sm" whiteSpace="pre-wrap">{value}</Text>
                            </Box>
                          ))}
                        </AccordionPanel>
                      </AccordionItem>
                    </Accordion>
                  )}
                </Flex>
              )}

              {/* Feedback Status */}
              <Flex justify="space-between" align="center" mt={2}>
                <Badge 
                  colorScheme={
                    conv.feedback_status === 'approved' ? 'green' : 
                    conv.feedback_status === 'pending' ? 'yellow' : 
                    'red'
                  }
                >
                  {conv.feedback_status || 'pending'}
                </Badge>
                <Button
                  size="xs"
                  variant="ghost"
                  colorScheme="red"
                  onClick={(e) => handleClearConversation(conv.id, e)}
                >
                  <IoTrash />
                </Button>
              </Flex>
            </Box>
          );
        })}
      </VStack>
    );
  };

  if (loading && threads.length === 0) {
    return (
      <Box p={4}>
        <Text>Loading conversations...</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={4}>
        <Text color="red.500">Error: {error}</Text>
      </Box>
    );
  }

  return (
    <Box p={4}>
      <Flex justify="space-between" align="center" mb={4}>
        <Heading size="md">Conversation History</Heading>
        <Button 
          leftIcon={<IoAdd />} 
          colorScheme="blue" 
          size="sm"
          onClick={onNewChat}
        >
          New Chat
        </Button>
      </Flex>
      
      <Divider mb={4} />
      
      <VStack spacing={3} align="stretch">
        {threads.length === 0 ? (
          <Text textAlign="center" color="gray.500">No conversations yet</Text>
        ) : (
          threads.map(thread => (
            <Card key={thread.thread_id} variant="outline" mb={2}>
              <CardBody p={3}>
                <Box>
                  <Flex justify="space-between" align="center" mb={2}>
                    <Text fontWeight="bold" fontSize="sm">Thread Started</Text>
                    <Text fontSize="xs" color="gray.500">
                      {new Date(thread.timestamp).toLocaleString()}
                    </Text>
                  </Flex>
                  
                  <Box 
                    bg="gray.50" 
                    p={2} 
                    borderRadius="md" 
                    mb={2}
                    cursor="pointer"
                    onClick={() => onSelectConversation(thread.first_conversation_id)}
                    _hover={{ bg: "gray.100" }}
                  >
                    <Text fontWeight="medium" noOfLines={2}>
                      {thread.preview}
                    </Text>
                  </Box>

                  {thread.conversation_count > 1 && (
                    <Accordion allowToggle>
                      <AccordionItem border="none">
                        <AccordionButton 
                          p={2} 
                          _hover={{ bg: "gray.50" }}
                          onClick={() => toggleThreadExpansion(thread.thread_id)}
                        >
                          <Box flex="1" textAlign="left" fontSize="sm">
                            <Badge colorScheme="blue" mr={2}>
                              {thread.conversation_count} messages
                            </Badge>
                            View Thread
                          </Box>
                          <AccordionIcon />
                        </AccordionButton>
                        <AccordionPanel pb={4}>
                          {threadConversations[thread.thread_id] ? (
                            renderThreadConversations(threadConversations[thread.thread_id])
                          ) : (
                            <Text fontSize="sm" color="gray.500" textAlign="center">
                              Loading conversations...
                            </Text>
                          )}
                        </AccordionPanel>
                      </AccordionItem>
                    </Accordion>
                  )}
                </Box>
              </CardBody>
            </Card>
          ))
        )}
      </VStack>
    </Box>
  );
};

export default ConversationHistory; 