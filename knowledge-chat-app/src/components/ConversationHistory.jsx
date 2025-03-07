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
        if (response.status === 'success') {
          setThreadConversations(prev => ({
            ...prev,
            [threadId]: response.conversations
          }));
          console.log(`Loaded conversations for thread ${threadId}:`, response.conversations);
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
                <Flex 
                  justify="space-between" 
                  align="center" 
                  onClick={() => onSelectConversation(thread.first_conversation_id)}
                  cursor="pointer"
                  _hover={{ bg: "gray.50" }}
                  p={2}
                  borderRadius="md"
                >
                  <Box>
                    <Text fontWeight="medium" noOfLines={1}>
                      {thread.preview}
                    </Text>
                    <Text fontSize="xs" color="gray.500">
                      {new Date(thread.timestamp).toLocaleString()}
                    </Text>
                    {thread.conversation_count > 1 && (
                      <Badge colorScheme="blue" mt={1}>
                        {thread.conversation_count} messages
                      </Badge>
                    )}
                  </Box>
                  <Button
                    size="sm"
                    variant="ghost"
                    colorScheme="red"
                    onClick={(e) => handleClearConversation(thread.first_conversation_id, e)}
                  >
                    <IoTrash />
                  </Button>
                </Flex>
                
                {thread.conversation_count > 1 && (
                  <Accordion allowToggle mt={2}>
                    <AccordionItem border="none">
                      <AccordionButton 
                        p={2} 
                        _hover={{ bg: "gray.50" }}
                        onClick={() => toggleThreadExpansion(thread.thread_id)}
                      >
                        <Box flex="1" textAlign="left" fontSize="sm">
                          Show all conversations in this thread
                        </Box>
                        <AccordionIcon />
                      </AccordionButton>
                      <AccordionPanel pb={4}>
                        {threadConversations[thread.thread_id] ? (
                          <VStack spacing={2} align="stretch">
                            {threadConversations[thread.thread_id].map(conv => (
                              <Flex
                                key={conv.id}
                                p={2}
                                borderRadius="md"
                                borderWidth="1px"
                                borderColor="gray.200"
                                justify="space-between"
                                align="center"
                                _hover={{ bg: "blue.50" }}
                                cursor="pointer"
                                onClick={() => onSelectConversation(conv.id)}
                              >
                                <Box>
                                  <Text fontSize="sm" fontWeight="medium" noOfLines={1}>
                                    {conv.preview}
                                  </Text>
                                  <Text fontSize="xs" color="gray.500">
                                    {new Date(conv.timestamp).toLocaleString()}
                                  </Text>
                                </Box>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  colorScheme="red"
                                  onClick={(e) => handleClearConversation(conv.id, e)}
                                >
                                  <IoTrash />
                                </Button>
                              </Flex>
                            ))}
                          </VStack>
                        ) : (
                          <Text fontSize="sm" color="gray.500" textAlign="center">
                            Loading conversations...
                          </Text>
                        )}
                      </AccordionPanel>
                    </AccordionItem>
                  </Accordion>
                )}
              </CardBody>
            </Card>
          ))
        )}
      </VStack>
    </Box>
  );
};

export default ConversationHistory; 