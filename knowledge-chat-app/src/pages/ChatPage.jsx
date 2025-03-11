import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Flex,
  VStack,
  HStack,
  Text,
  Input,
  Button,
  Avatar,
  Divider,
  Card,
  CardBody,
  CardHeader,
  IconButton,
  useColorModeValue,
  Heading,
  Textarea,
  InputGroup,
  InputRightElement,
  Badge,
  Code,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Tag,
  TagLabel,
  useDisclosure,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Select,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Progress,
  UnorderedList,
  ListItem,
  SimpleGrid,
  useToast
} from '@chakra-ui/react'
import { 
  IoSend, 
  IoAdd, 
  IoMenu, 
  IoInformationCircle, 
  IoDocumentText, 
  IoChevronDown,
  IoServer,
  IoAnalytics,
  IoCodeSlash,
  IoGrid,
  IoLayers,
  IoExpand,
  IoContract,
  IoCheckmark,
  IoClose
} from 'react-icons/io5'
import { useParams, useNavigate } from 'react-router-dom'

// Sample chat history data
const chatHistory = [
  {
    id: 1,
    title: "Customer Data Schema",
    preview: "Tell me about the customer data schema",
    timestamp: "2023-11-15T10:30:00Z",
    active: true
  },
  {
    id: 2,
    title: "SQL Query Optimization",
    preview: "How can I optimize this SQL query?",
    timestamp: "2023-11-14T14:45:00Z",
    active: false
  },
  {
    id: 3,
    title: "ETL Pipeline Analysis",
    preview: "Explain our ETL pipeline",
    timestamp: "2023-11-10T09:15:00Z",
    active: false
  }
];

// Sample conversation data for the active chat
const sampleConversation = [
  {
    id: 1,
    role: 'assistant',
    content: "Hello! I'm your Data Architecture Assistant. How can I help you today? You can ask me about database schemas, data models, ETL processes, or query optimization.",
    timestamp: "2023-11-15T10:30:00Z"
  },
  {
    id: 2,
    role: 'user',
    content: "Tell me about the customer data schema",
    timestamp: "2023-11-15T10:31:00Z"
  },
  {
    id: 3,
    role: 'assistant',
    content: "The customer data schema consists of several related tables that store information about customers and their interactions with our platform.",
    timestamp: "2023-11-15T10:31:30Z",
    sources: [
      { title: 'Data Dictionary', type: 'document' },
      { title: 'Snowflake Schema', type: 'database' }
    ],
    tables: [
      { name: 'customers', rowCount: '1.2M', lastUpdated: 'Today, 09:15 AM' },
      { name: 'customer_addresses', rowCount: '1.5M', lastUpdated: 'Today, 09:15 AM' },
      { name: 'customer_preferences', rowCount: '950K', lastUpdated: 'Yesterday, 11:30 PM' }
    ]
  }
];

// Update the FormattedMessage component to better handle data architect responses

const FormattedMessage = ({ content }) => {
  // Handle case where content is not a string
  if (typeof content !== 'string') {
    try {
      content = JSON.stringify(content, null, 2);
    } catch (e) {
      content = "Error displaying content";
    }
  }
  
  // Check if content is empty
  if (!content || content.trim() === '') {
    return <Text color="gray.500">No content available</Text>;
  }
  
  // Check if content contains markdown sections (## or **)
  const hasMarkdown = content.includes('##') || content.includes('**');
  
  if (hasMarkdown) {
    // Split by markdown headers (##)
    const sections = content.split(/##\s+/);
    
    return (
      <VStack align="start" spacing={4} width="100%">
        {sections.map((section, idx) => {
          if (idx === 0 && !section.trim()) return null;
          
          if (idx === 0) {
            // This is the intro text before any headers
            return (
              <Text 
                key={idx}
                fontSize="16px"
                fontFamily="'Merriweather', Georgia, serif"
                lineHeight="1.7"
                color="gray.800"
                whiteSpace="pre-wrap"
              >
                {section}
              </Text>
            );
          }
          
          // For sections with headers
          const sectionParts = section.split(/\n/);
          const sectionTitle = sectionParts[0];
          const sectionContent = sectionParts.slice(1).join('\n');
          
          return (
            <Box key={idx} width="100%" mt={2}>
              <Heading 
                size="md" 
                color="purple.700"
                fontWeight="600"
                fontFamily="'Playfair Display', Georgia, serif"
                pb={2}
                borderBottom="2px solid"
                borderColor="purple.200"
                width="fit-content"
                fontSize="18px"
                letterSpacing="0.02em"
                mb={3}
              >
                {sectionTitle}
              </Heading>
              <Text 
                fontSize="16px"
                fontFamily="'Merriweather', Georgia, serif"
                lineHeight="1.7"
                color="gray.800"
                whiteSpace="pre-wrap"
                pl={2}
                borderLeft="3px solid"
                borderColor="purple.100"
              >
                {sectionContent}
              </Text>
            </Box>
          );
        })}
      </VStack>
    );
  } else {
    // Simple text display for non-markdown content
    return (
      <Text 
        fontSize="16px"
        fontFamily="'Merriweather', Georgia, serif"
        lineHeight="1.7"
        color="gray.800"
        whiteSpace="pre-wrap"
      >
        {content}
      </Text>
    );
  }
};

const ChatMessage = ({ message, onFeedbackSubmit }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Format the message content sections
  const formatMessageContent = () => {
    const details = message.details?.parsed_question;
    if (!details) return null;

    return (
      <VStack align="start" spacing={3} width="100%">
        {/* Business Understanding */}
        <Box>
          <Text fontWeight="medium">Business Understanding:</Text>
          <Text>{details.rephrased_question}</Text>
        </Box>

        {/* Business Context */}
        <Box>
          <Text fontWeight="medium">Business Context:</Text>
          <UnorderedList>
            <ListItem><strong>Domain:</strong> {details.business_context.domain}</ListItem>
            <ListItem><strong>Objective:</strong> {details.business_context.primary_objective}</ListItem>
            <ListItem><strong>Key Entities:</strong> {details.business_context.key_entities.join(', ')}</ListItem>
            <ListItem><strong>Impact:</strong> {details.business_context.business_impact}</ListItem>
          </UnorderedList>
        </Box>

        {/* Key Points */}
        <Box>
          <Text fontWeight="medium">Key Business Points:</Text>
          <UnorderedList>
            {details.key_points.map((point, idx) => (
              <ListItem key={idx}>{point}</ListItem>
            ))}
          </UnorderedList>
        </Box>

        {/* Assumptions */}
        <Box>
          <Text fontWeight="medium">Assumptions to Verify:</Text>
          <UnorderedList>
            {details.assumptions.map((assumption, idx) => (
              <ListItem key={idx}>{assumption}</ListItem>
            ))}
          </UnorderedList>
        </Box>

        {/* Clarifying Questions */}
        <Box>
          <Text fontWeight="medium">Clarifying Questions:</Text>
          <UnorderedList>
            {details.clarifying_questions.map((question, idx) => (
              <ListItem key={idx}>{question}</ListItem>
            ))}
          </UnorderedList>
        </Box>
      </VStack>
    );
  };

  return (
    <VStack align="stretch" spacing={4} w="100%">
      {/* Always show the message content */}
      <Box bg="white" p={4} borderRadius="md" shadow="sm">
        <Text whiteSpace="pre-wrap">{message.content}</Text>
      </Box>

      {/* View Details Accordion */}
      <Accordion allowToggle width="100%">
        <AccordionItem border="none">
          <AccordionButton 
            px={4} 
            py={2}
            bg="gray.50"
            _hover={{ bg: 'gray.100' }}
            borderRadius="md"
          >
            <Box flex="1" textAlign="left">
              <Text fontWeight="medium" color="blue.600">View Details</Text>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            {formatMessageContent()}
          </AccordionPanel>
        </AccordionItem>
      </Accordion>

      {/* Show feedback section if feedback is required */}
      {message.details?.requires_confirmation && (
        <InlineFeedback message={message} onFeedbackSubmit={onFeedbackSubmit} />
      )}
    </VStack>
  );
};

// Add a debug function to log message details
const debugMessageDetails = (message) => {
  console.log("Message details for debugging:");
  console.log("- feedback_id:", message.details?.feedback_id);
  console.log("- conversation_id:", message.details?.conversation_id);
  console.log("- Full details:", message.details);
};

// Update the InlineFeedback component
const InlineFeedback = ({ message, onFeedbackSubmit }) => {
  const [comments, setComments] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const toast = useToast();

  // Debug the message details when component mounts
  useEffect(() => {
    debugMessageDetails(message);
  }, [message]);

  const handleSubmit = async (isApproved) => {
    if (isSubmitting) return;
    
    console.log("Feedback button clicked:", isApproved ? "Confirm" : "Need Clarification");
    debugMessageDetails(message);
    
    // Use conversation_id as feedback_id if feedback_id is missing
    const feedback_id = message.details?.feedback_id || message.details?.conversation_id;
    const conversation_id = message.details?.conversation_id;
    
    // Check if we have the required fields
    if (!feedback_id) {
      console.error("Missing feedback_id and no fallback available");
      toast({
        title: 'Error',
        description: 'Missing feedback ID information',
        status: 'error',
        duration: 3000,
      });
      return;
    }
    
    if (!conversation_id) {
      console.error("Missing conversation_id in message details");
      toast({
        title: 'Error',
        description: 'Missing conversation ID information',
        status: 'error',
        duration: 3000,
      });
      return;
    }
    
    setIsSubmitting(true);
    try {
      // Try a direct fetch to the feedback endpoint
      const feedbackPayload = {
        feedback_id: feedback_id,
        conversation_id: conversation_id,
        approved: isApproved,
        comments: comments || null
      };
      
      console.log("Sending direct feedback payload:", feedbackPayload);
      
      // Make a direct fetch request
      const response = await fetch('http://localhost:8000/feedback/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackPayload)
      });
      
      console.log("Direct fetch response status:", response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        throw new Error(`Failed to submit feedback: ${errorText}`);
      }
      
      // Success - clear comments and show toast
      setComments('');
      toast({
        title: isApproved ? 'Understanding Confirmed' : 'Clarification Requested',
        description: 'Your feedback has been submitted',
        status: 'success',
        duration: 3000,
      });
      
      // Call the parent handler to update UI
      onFeedbackSubmit({
        feedback_id: feedback_id,
        conversation_id: conversation_id,
        approved: isApproved,
        comments: comments
      });
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <VStack align="stretch" spacing={3}>
      {/* Add debug info in development */}
      {process.env.NODE_ENV !== 'production' && (
        <Box p={2} bg="gray.100" fontSize="xs" mt={2}>
          <Text>Debug Info:</Text>
          <Text>feedback_id: {message.details?.feedback_id || 'missing'}</Text>
          <Text>conversation_id: {message.details?.conversation_id || 'missing'}</Text>
          <Text>Using feedback_id: {message.details?.feedback_id || message.details?.conversation_id || 'none available'}</Text>
        </Box>
      )}
      
      {/* Feedback Section */}
      <Textarea
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        placeholder="Add any clarifications or corrections to the business understanding..."
        bg="white"
        size="sm"
      />

      <HStack spacing={4}>
        <Button
          colorScheme="green"
          leftIcon={<IoCheckmark />}
          onClick={() => handleSubmit(true)}
          isLoading={isSubmitting}
        >
          Confirm Understanding
        </Button>
        <Button
          colorScheme="blue"
          leftIcon={<IoClose />}
          onClick={() => handleSubmit(false)}
          isLoading={isSubmitting}
        >
          Need Clarification
        </Button>
      </HStack>
    </VStack>
  );
};

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [processingStep, setProcessingStep] = useState('');
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const messagesEndRef = useRef(null);
  const [currentConversationId, setCurrentConversationId] = useState(conversationId || null);
  const [currentThreadId, setCurrentThreadId] = useState(null);
  const [feedbackRequired, setFeedbackRequired] = useState(false);
  const [feedbackData, setFeedbackData] = useState(null);

  // Initialize or load conversation from ID
  useEffect(() => {
    // Clear messages when component mounts if there's no conversation ID
    if (!conversationId) {
      setMessages([]);
      setActiveConversationId(null);
      setCurrentConversationId(null);
      setCurrentThreadId(null);
    } else {
      // If we have a conversation ID on mount, fetch that conversation
      fetchConversation(conversationId);
    }
  }, [conversationId]);

  // Fetch conversation history
  const fetchConversations = async () => {
    try {
      setIsHistoryLoading(true);
      const response = await fetch('http://localhost:8000/api/conversations');
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }
      const data = await response.json();
      if (data.status === 'success') {
        setConversations(data.conversations);
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      toast({
        title: 'Error',
        description: 'Failed to load conversation history',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setIsHistoryLoading(false);
    }
  };

  // Fetch a specific conversation
  const fetchConversation = async (id) => {
    try {
      setLoading(true);
      setActiveConversationId(id);
      
      const response = await fetch(`http://localhost:8000/api/chat/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch conversation');
      }
      
      const data = await response.json();
      console.log("Fetched conversation:", data);
      
      // Format messages for display
      const formattedMessages = [];
      
      if (data.messages && Array.isArray(data.messages)) {
        data.messages.forEach(msg => {
          formattedMessages.push({
            id: msg.id || formattedMessages.length,
            role: msg.role || 'assistant',
            content: msg.content || '',
            details: msg.details || {},
          });
        });
      }
      
      setMessages(formattedMessages);
      setCurrentConversationId(id);
      setCurrentThreadId(data.thread_id || null);
      
    } catch (error) {
      console.error('Error fetching conversation:', error);
      toast({
        title: 'Error',
        description: 'Failed to load conversation',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setLoading(false);
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Start a new conversation
  const startNewConversation = () => {
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    setCurrentThreadId(null);
    setFeedbackRequired(false);
    setFeedbackData(null);
    navigate('/chat', { replace: true });
  };

  // Handle conversation selection
  const handleConversationSelect = (id) => {
    fetchConversation(id);
    onClose(); // Close the drawer on mobile
  };

  // Add this function to handle architect response
  const handleArchitectResponse = async (conversationId) => {
    try {
      setProcessingStep('Generating detailed data architecture analysis...');
      
      const response = await fetch(`http://localhost:8000/chat/architect/${conversationId}`);
      
      if (!response.ok) {
        throw new Error('Failed to generate architect response');
      }
      
      const data = await response.json();
      console.log("Architect response:", data);
      
      if (data.status === 'success') {
        // Add the architect's response to the chat
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.response,
          id: `architect-${prev.length}`,
          details: {
            is_architect_response: true,
            conversation_id: conversationId,
            schema_results: data.schema_results,
            code_results: data.code_results,
            sections: data.sections
          }
        }]);
        
        toast({
          title: 'Analysis Complete',
          description: 'Data architecture analysis has been generated',
          status: 'success',
          duration: 3000,
        });
      }
    } catch (error) {
      console.error('Error generating architect response:', error);
      toast({
        title: 'Error',
        description: 'Failed to generate detailed analysis',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setProcessingStep('');
    }
  };

  // Update the handleFeedbackSubmit function to include architect response generation
  const handleFeedbackSubmit = async (feedback) => {
    console.log("Submitting feedback:", feedback);
    
    try {
      const response = await fetch('http://localhost:8000/feedback/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback)
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }
      
      const data = await response.json();
      console.log("Feedback response:", data);
      
      toast({
        title: 'Feedback Submitted',
        description: 'Thank you for your feedback',
        status: 'success',
        duration: 3000,
      });
      
      // Update messages to reflect feedback status
      setMessages(prev => prev.map(message => {
        if (message.details?.feedback_id === feedback.feedback_id) {
          return {
            ...message,
            details: {
              ...message.details,
              feedback_status: feedback.approved ? 'approved' : 'needs_improvement'
            }
          };
        }
        return message;
      }));
      
      // If feedback was approved, generate architect response
      if (feedback.approved) {
        setTimeout(() => {
          handleArchitectResponse(feedback.conversation_id);
        }, 500);
      }
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast({
        title: 'Error',
        description: 'Failed to submit feedback',
        status: 'error',
        duration: 3000,
      });
    }
  };

  // Update the sendMessage function to better handle processing steps
  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = input;
    setInput('');
    
    // Add user message to the chat
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: userMessage,
      id: prev.length
    }]);
    
    // Set loading state
    setLoading(true);
    setProcessingStep('Analyzing your question...');
    
    try {
      // Prepare the chat request
      const chatRequest = {
        message: userMessage,
        conversation_id: currentConversationId,
        thread_id: currentThreadId
      };
      
      // Step 1: Initial processing
      setProcessingStep('Processing your request...');
      
      // API call to submit the message
      const response = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(chatRequest)
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Chat response:", data);
      
      // Step 2: Parse and understand
      setProcessingStep('Understanding business context...');
      
      // Add the assistant's response to the chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        id: prev.length,
        details: {
          conversation_id: data.conversation_id,
          feedback_id: data.feedback_id,
          feedback_required: data.feedback_required,
          feedback_status: data.feedback_status,
          parsed_question: data.parsed_question,
          requires_confirmation: data.feedback_required,
        }
      }]);
      
      // Update conversation tracking info
      setCurrentConversationId(data.conversation_id);
      if (data.thread_id) {
        setCurrentThreadId(data.thread_id);
      }
      
      // Handle any feedback requirements
      if (data.feedback_required) {
        setFeedbackRequired(true);
        setFeedbackData({
          feedback_id: data.feedback_id,
          conversation_id: data.conversation_id
        });
      }
      
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: 'Error',
        description: 'Failed to send message. Please try again.',
        status: 'error',
        duration: 3000,
      });
      
      // Add error message to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        id: prev.length,
        isError: true
      }]);
    } finally {
      setLoading(false);
      setProcessingStep('');
    }
  };

  // Render a message
  const renderMessage = (message) => {
    const isUser = message.role === 'user';
    const isArchitectResponse = message.details?.is_architect_response;
    
    return (
      <Box 
        key={message.id}
        bg={isUser ? 'blue.50' : isArchitectResponse ? 'purple.50' : 'white'}
        p={4}
        borderRadius="lg"
        borderWidth="1px"
        borderColor={
          isUser ? 'blue.200' : 
          isArchitectResponse ? 'purple.200' : 
          'gray.200'
        }
        maxW={isArchitectResponse ? "95%" : "90%"}
        alignSelf={isUser ? 'flex-end' : 'flex-start'}
        boxShadow="sm"
        mb={4}
        width={isArchitectResponse ? "95%" : "auto"}
      >
        <VStack align="stretch" spacing={3}>
          <HStack>
            <Avatar 
              size="sm" 
              bg={
                isUser ? 'blue.500' : 
                isArchitectResponse ? 'purple.500' : 
                'green.500'
              } 
              name={
                isUser ? 'You' : 
                isArchitectResponse ? 'Data Architect' : 
                'Assistant'
              } 
            />
            <Text fontWeight="bold" color={
              isUser ? 'blue.700' : 
              isArchitectResponse ? 'purple.700' : 
              'green.700'
            }>
              {isUser ? 'You' : isArchitectResponse ? 'Data Architect' : 'Assistant'}
            </Text>
            
            {message.details?.feedback_status && (
              <Badge 
                colorScheme={
                  message.details.feedback_status === 'approved' ? 'green' : 
                  message.details.feedback_status === 'needs_improvement' ? 'orange' : 
                  'blue'
                }
              >
                {message.details.feedback_status === 'approved' ? 'Approved' : 
                message.details.feedback_status === 'needs_improvement' ? 'Needs Review' : 
                'Pending'}
              </Badge>
            )}
          </HStack>
          
          <Text whiteSpace="pre-wrap">{message.content}</Text>
          
          {/* If message has details with feedback required, show feedback UI */}
          {message.details?.requires_confirmation && (
            <Box mt={3} p={3} bg="gray.50" borderRadius="md">
              <Text fontWeight="medium" mb={2}>
                Is this response helpful?
              </Text>
              <HStack>
                <Button 
                  colorScheme="green" 
                  size="sm" 
                  onClick={() => handleFeedbackSubmit({
                    feedback_id: message.details.feedback_id,
                    conversation_id: message.details.conversation_id,
                    approved: true,
                    comments: "User approved"
                  })}
                >
                  Yes, this is helpful
                </Button>
                <Button 
                  colorScheme="orange" 
                  size="sm"
                  onClick={() => handleFeedbackSubmit({
                    feedback_id: message.details.feedback_id,
                    conversation_id: message.details.conversation_id,
                    approved: false,
                    comments: "User requested improvements"
                  })}
                >
                  No, needs improvement
                </Button>
              </HStack>
            </Box>
          )}
          
          {/* If the message has parsed question details, show them in an accordion */}
          {message.details?.parsed_question && (
            <Accordion allowToggle>
              <AccordionItem border="none">
                <AccordionButton px={0} _hover={{ bg: 'transparent' }}>
                  <Text fontSize="sm" color="blue.500">Show analysis details</Text>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} px={0}>
                  <Code p={2} fontSize="xs" variant="subtle" borderRadius="md" whiteSpace="pre-wrap">
                    {JSON.stringify(message.details.parsed_question, null, 2)}
                  </Code>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}
          
          {/* If message is an architect response with sections, show them */}
          {isArchitectResponse && message.details?.sections && (
            <Accordion allowToggle width="100%" defaultIndex={[0]}>
              <AccordionItem border="none" borderTop="1px solid" borderColor="purple.200">
                <AccordionButton py={3} _hover={{ bg: "purple.50" }}>
                  <Box flex="1" textAlign="left">
                    <Text fontSize="15px" color="purple.600" fontWeight="500">
                      View Technical Implementation Details
                    </Text>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} pt={2}>
                  <VStack align="stretch" spacing={4}>
                    {Object.entries(message.details.sections).map(([sectionName, sectionContent], idx) => (
                      <Box key={idx}>
                        <Text fontWeight="bold" mb={2} color="purple.700">{sectionName}:</Text>
                        <Box 
                          pl={4} 
                          borderLeft="2px solid" 
                          borderColor="purple.100"
                          fontSize="15px"
                          lineHeight="1.6"
                          whiteSpace="pre-wrap"
                        >
                          {sectionContent}
                        </Box>
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}
          
          {/* For architect responses, show schema and code results */}
          {isArchitectResponse && (message.details?.schema_results || message.details?.code_results) && (
            <Accordion allowToggle width="100%">
              <AccordionItem border="none" borderTop="1px solid" borderColor="purple.200">
                <AccordionButton py={3} _hover={{ bg: "purple.50" }}>
                  <Box flex="1" textAlign="left">
                    <Text fontSize="15px" color="purple.600" fontWeight="500">
                      View Referenced Data & Code
                    </Text>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} pt={2}>
                  {message.details.schema_results && message.details.schema_results.length > 0 && (
                    <Box mb={4}>
                      <Text fontWeight="bold" mb={2} color="purple.700">Database Schemas:</Text>
                      <VStack align="stretch" spacing={2}>
                        {message.details.schema_results.map((schema, idx) => (
                          <Box 
                            key={idx} 
                            p={3} 
                            bg="gray.50" 
                            borderRadius="md" 
                            borderLeft="3px solid" 
                            borderColor="purple.300"
                          >
                            <Text fontWeight="medium">{schema.table_name || schema.name}</Text>
                            {schema.description && (
                              <Text fontSize="sm" color="gray.600" mt={1}>{schema.description}</Text>
                            )}
                          </Box>
                        ))}
                      </VStack>
                    </Box>
                  )}
                  
                  {message.details.code_results && message.details.code_results.length > 0 && (
                    <Box>
                      <Text fontWeight="bold" mb={2} color="purple.700">Code References:</Text>
                      <VStack align="stretch" spacing={2}>
                        {message.details.code_results.map((code, idx) => (
                          <Box 
                            key={idx} 
                            p={3} 
                            bg="gray.50" 
                            borderRadius="md" 
                            borderLeft="3px solid" 
                            borderColor="blue.300"
                          >
                            <Text fontWeight="medium">{code.file_path || code.name}</Text>
                            {code.snippet && (
                              <Code p={2} mt={2} fontSize="xs" overflowX="auto" whiteSpace="pre">
                                {code.snippet}
                              </Code>
                            )}
                          </Box>
                        ))}
                      </VStack>
                    </Box>
                  )}
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}
        </VStack>
      </Box>
    );
  };

  // Clear the current chat
  const clearChat = () => {
    // Save current conversation ID before clearing
    const currentId = activeConversationId;
    
    // Clear messages and conversation ID
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    setCurrentThreadId(null);
    
    toast({
      title: 'Chat Cleared',
      description: 'The chat has been cleared. You can find it in the history tab.',
      status: 'info',
      duration: 3000
    });
  };

  // Debug message changes
  useEffect(() => {
    console.log("Messages state changed:", messages);
  }, [messages]);

  return (
    <Box height="100vh" overflow="hidden">
      {/* Mobile drawer for conversation history */}
      <Drawer isOpen={isOpen} placement="left" onClose={onClose}>
        <DrawerOverlay />
        <DrawerContent>
          <DrawerCloseButton />
          <DrawerHeader borderBottomWidth="1px">Conversation History</DrawerHeader>
          <DrawerBody>
            {/* Render conversation history sidebar */}
            <VStack align="stretch" spacing={3} w="100%" p={3}>
              <Button 
                leftIcon={<IoAdd />} 
                colorScheme="blue" 
                onClick={startNewConversation}
                mb={4}
              >
                New Conversation
              </Button>
              
              <Divider mb={2} />
              
              {isHistoryLoading ? (
                <Progress size="xs" isIndeterminate colorScheme="blue" />
              ) : conversations.length === 0 ? (
                <Text color="gray.500" textAlign="center" py={4}>No conversation history</Text>
              ) : (
                <VStack align="stretch" spacing={2}>
                  {conversations
                    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)) // Sort by timestamp, newest first
                    .map((conv) => (
                      <Box 
                        key={conv.id}
                        p={3}
                        borderRadius="md"
                        bg={activeConversationId === conv.id ? "blue.50" : "white"}
                        borderWidth="1px"
                        borderColor={activeConversationId === conv.id ? "blue.200" : "gray.200"}
                        cursor="pointer"
                        onClick={() => handleConversationSelect(conv.id)}
                        _hover={{ 
                          bg: activeConversationId === conv.id ? "blue.50" : "gray.50",
                          borderColor: "blue.300"
                        }}
                        transition="all 0.2s"
                      >
                        <HStack justify="space-between" mb={1}>
                          <Text 
                            fontSize="sm" 
                            color="gray.500"
                            isTruncated
                          >
                            {new Date(conv.timestamp).toLocaleString()}
                          </Text>
                          {/* Only show badge for approved or needs improvement status */}
                          {(conv.feedback_status === 'approved' || conv.feedback_status === 'needs_improvement') && (
                            <Badge 
                              colorScheme={
                                conv.feedback_status === 'approved' ? 'green' : 'orange'
                              }
                              fontSize="xs"
                            >
                              {conv.feedback_status === 'approved' ? 'Approved' : 'Needs Review'}
                            </Badge>
                          )}
                        </HStack>
                        <Text 
                          fontWeight="medium" 
                          isTruncated
                          color={activeConversationId === conv.id ? "blue.700" : "gray.800"}
                        >
                          {conv.preview}
                        </Text>
                      </Box>
                    ))}
                </VStack>
              )}
            </VStack>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
      
      <Flex h="100%" direction="column">
        {/* Header with mobile menu */}
        <Flex 
          p={4} 
          borderBottomWidth="1px" 
          borderColor="gray.200" 
          align="center"
          justify="space-between"
          bg="white"
        >
          <HStack>
            <IconButton
              icon={<IoMenu />}
              aria-label="Open menu"
              display={{ base: "flex", md: "none" }}
              mr={3}
              onClick={onOpen}
            />
            <Heading size="md">
              {currentConversationId ? "Conversation" : "New Chat"}
            </Heading>
          </HStack>
          
          <Button
            leftIcon={<IoClose />}
            variant="ghost"
            colorScheme="blue"
            size="sm"
            onClick={clearChat}
            isDisabled={messages.length === 0}
          >
            Clear Chat
          </Button>
        </Flex>
        
        {/* Input area at the top */}
        <Box 
          p={4} 
          borderBottomWidth="1px" 
          borderColor="gray.200"
          bg="white"
        >
          <HStack>
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about database schemas, data models, or SQL queries..."
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              disabled={loading}
              size="lg"
              py={6}
              borderRadius="md"
              _focus={{
                borderColor: "blue.400",
                boxShadow: "0 0 0 1px blue.400"
              }}
            />
            <Button 
              onClick={sendMessage} 
              isLoading={loading}
              colorScheme="blue"
              size="lg"
              px={8}
              height="56px"
              leftIcon={<IoSend />}
            >
              Send
            </Button>
          </HStack>
        </Box>
        
        {/* Messages area below input */}
        <Box 
          flex="1" 
          overflowY="auto" 
          p={6} 
          bg="gray.50"
        >
          <VStack spacing={6} mb={8} align="stretch">
            {messages.map((message) => renderMessage(message))}
            
            {processingStep && (
              <Box 
                p={4} 
                bg="blue.50" 
                borderRadius="md" 
                width={["98%", "95%", "90%"]}
                alignSelf="flex-start"
                boxShadow="0 2px 8px rgba(0, 0, 0, 0.05)"
                borderWidth="1px"
                borderColor="blue.100"
              >
                <HStack>
                  <Box as={IoAnalytics} color="blue.500" boxSize={5} mr={2} />
                  <Text fontWeight="500" color="blue.700">{processingStep}</Text>
                </HStack>
                <Progress size="xs" colorScheme="blue" isIndeterminate mt={3} />
              </Box>
            )}
            <div ref={messagesEndRef} />
          </VStack>
        </Box>
      </Flex>
    </Box>
  );
};

export default ChatPage; 