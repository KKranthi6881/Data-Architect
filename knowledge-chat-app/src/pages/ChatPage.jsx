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
  const [activeTab, setActiveTab] = useState('chat');
  const [currentConversationId, setCurrentConversationId] = useState(conversationId || null);
  const [currentThreadId, setCurrentThreadId] = useState(null);
  const [feedbackRequired, setFeedbackRequired] = useState(false);
  const [feedbackData, setFeedbackData] = useState(null);

  // Add this to the top of the component
  useEffect(() => {
    console.log("Current messages:", messages);
  }, [messages]);

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
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch conversation');
      }
      const data = await response.json();
      console.log("API Response:", data); // Log the full API response
      
      if (data.status === 'success') {
        const conversation = data.conversation;
        console.log("Conversation data:", conversation); // Log the conversation data
        
        // Clear existing messages
        setMessages([]);
        
        // Add user message
        const userMessage = {
          type: 'user',
          content: conversation.query || "No query available",
          id: `${id}-query`
        };
        console.log("Adding user message:", userMessage);
        
        // Add assistant response with proper formatting
        const assistantMessage = {
          type: 'assistant',
          content: typeof conversation.response === 'string' 
            ? conversation.response 
            : JSON.stringify(conversation.response),
          id: `${id}-response`,
          details: {
            conversation_id: id,
            feedback_status: conversation.feedback?.status || 'pending',
            parsed_question: conversation.technical_details, // Use technical_details for parsed_question
            sources: conversation.context
          }
        };
        console.log("Adding assistant message:", assistantMessage);
        
        // Update messages state with both messages
        setMessages([userMessage, assistantMessage]);
        
        setActiveConversationId(id);
        
        // Update URL without reloading
        navigate(`/chat/${id}`, { replace: true });
        
        // Set thread ID if available in the history
        if (conversation.thread_id) {
          setCurrentThreadId(conversation.thread_id);
        }
      }
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

  // Load conversation from URL parameter
  useEffect(() => {
    if (conversationId) {
      fetchConversation(conversationId);
    }
  }, [conversationId]);

  // Fetch conversation history on component mount
  useEffect(() => {
    fetchConversations();
  }, []);

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

  const analyzeQuestion = async (question) => {
    try {
      setProcessingStep('Analyzing your question...');
      const response = await fetch('/api/analyze-question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      
      const analysis = await response.json();
      setAnalysisResult(analysis);
      setProcessingStep('');
    } catch (error) {
      console.error('Error analyzing question:', error);
      setProcessingStep('');
    }
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const userQuestion = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQuestion }]);
    await analyzeQuestion(userQuestion);
  };

  const handleAnalysisConfirmation = async (approved) => {
    if (approved) {
      // Proceed with the rephrased question
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `I'll answer based on this interpretation: ${analysisResult.rephrased_question}`
      }]);
      // Call your existing question processing logic here
    } else {
      // Ask user to rephrase or provide clarification
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Could you please rephrase your question or provide more details?"
      }]);
    }
    setAnalysisResult(null);
  };

  const renderAnalysisConfirmation = () => {
    if (!analysisResult) return null;

    return (
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 mb-4">
        <h3 className="font-medium text-blue-800 mb-2">I understand your question as:</h3>
        <p className="text-gray-800 mb-3">{analysisResult.rephrased_question}</p>
        
        {analysisResult.assumptions.length > 0 && (
          <div className="mb-3">
            <h4 className="font-medium text-blue-800">Assumptions:</h4>
            <ul className="list-disc list-inside text-gray-700">
              {analysisResult.assumptions.map((assumption, i) => (
                <li key={i}>{assumption}</li>
              ))}
            </ul>
          </div>
        )}

        {analysisResult.follow_up_questions.length > 0 && (
          <div className="mb-3">
            <h4 className="font-medium text-blue-800">You might also want to know:</h4>
            <ul className="list-disc list-inside text-gray-700">
              {analysisResult.follow_up_questions.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex space-x-3 mt-4">
          <button
            onClick={() => handleAnalysisConfirmation(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Yes, that's correct
          </button>
          <button
            onClick={() => handleAnalysisConfirmation(false)}
            className="bg-gray-200 text-gray-800 px-4 py-2 rounded hover:bg-gray-300"
          >
            No, let me rephrase
          </button>
        </div>
      </div>
    );
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    const userMessage = input;
    setInput('');
    
    // Add user message
    setMessages(prev => [...prev, { 
      type: 'user', 
      content: userMessage 
    }]);

    try {
      // Show processing steps
      setProcessingStep('Analyzing code and documentation...');
      
      const response = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: currentConversationId || null,
          thread_id: currentThreadId
        })
      });

      const data = await response.json();
      console.log("Response from server:", data);

      // Make sure we have a conversation ID
      if (data.conversation_id) {
        setCurrentConversationId(data.conversation_id);
      }

      // Add assistant response with all required fields
      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        details: {
          ...data,
          conversation_id: data.conversation_id,
          feedback_id: data.feedback_id || data.conversation_id, // Use conversation_id as fallback
          feedback_status: 'pending',
          requires_confirmation: true, // Always show feedback options for new messages
          thread_id: data.thread_id
        }
      };
      
      console.log("Adding assistant message with details:", assistantMessage.details);
      setMessages(prev => [...prev, assistantMessage]);

      // Set thread ID from response
      if (data.thread_id) {
        setCurrentThreadId(data.thread_id);
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Error processing your request'
      }]);
    }

    setProcessingStep('');
    setLoading(false);
  };

  const handleFeedbackSubmit = async (feedback) => {
    try {
      setLoading(true);
      console.log("Starting feedback submission process");

      const feedbackPayload = {
        feedback_id: feedback.feedback_id,
        conversation_id: feedback.conversation_id,
        approved: feedback.approved,
        comments: feedback.comments || null
      };

      // First API call - submit feedback
      const response = await fetch('http://localhost:8000/feedback/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackPayload)
      });

      if (!response.ok) {
        throw new Error(`Failed to submit feedback: ${await response.text()}`);
      }

      const responseData = await response.json();
      console.log("Feedback API response:", responseData);

      if (feedback.approved) {
        // Update original message to show approved status
        setMessages(prev => prev.map(msg => 
          msg.details?.feedback_id === feedback.feedback_id
            ? {
                ...msg,
                details: {
                  ...msg.details,
                  feedback_status: 'approved',
                  requires_confirmation: false
                }
              }
            : msg
        ));

        // Add loading message
        const loadingMessage = {
          id: Date.now(),
          type: 'assistant',
          content: "Preparing detailed data architect analysis...",
          details: {
            is_architect_response: true,
            is_loading: true
          }
        };
        
        setMessages(prev => [...prev, loadingMessage]);

        // Second API call - fetch architect response
        const architectResponse = await fetch(`http://localhost:8000/chat/architect/${feedback.conversation_id}`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!architectResponse.ok) {
          throw new Error('Failed to fetch architect response');
        }

        const architectData = await architectResponse.json();
        console.log("Architect API response:", architectData);

        // Create architect message
        const architectMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: architectData.response || architectData.chat_response,
          details: {
            conversation_id: feedback.conversation_id,
            is_architect_response: true,
            feedback_status: 'final',
            requires_confirmation: false,
            sections: architectData.sections || {}
          }
        };

        // Update messages - remove loading and add architect response
        setMessages(prev => {
          const filteredMessages = prev.filter(msg => !msg.details?.is_loading);
          console.log("Messages before adding architect response:", filteredMessages);
          const newMessages = [...filteredMessages, architectMessage];
          console.log("Messages after adding architect response:", newMessages);
          return newMessages;
        });

        // Force scroll to bottom
        setTimeout(() => {
          if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
          }
        }, 100);

        toast({
          title: 'Analysis Complete',
          description: 'The data architect has provided a detailed solution.',
          status: 'success',
          duration: 3000
        });
      }

      await fetchConversations();

    } catch (error) {
      console.error('Error in handleFeedbackSubmit:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };

  // Update message display with wider boxes
  const renderMessage = (message) => {
    const requiresFeedback = message.type === 'assistant' && 
                            message.details?.requires_confirmation === true && 
                            message.details?.feedback_status !== 'approved' && 
                            message.details?.feedback_status !== 'needs_improvement';
    
    const isArchitectResponse = message.type === 'assistant' && 
                               message.details?.is_architect_response === true;
    
    const isLoading = message.details?.is_loading === true;
    
    return (
      <Box 
        key={message.id}
        bg={message.type === 'user' ? 'blue.50' : isArchitectResponse ? 'purple.50' : 'white'}
        p={5}
        borderRadius="lg"
        alignSelf="flex-start"
        width={["98%", "95%", "90%"]}
        boxShadow="0 2px 8px rgba(0, 0, 0, 0.08)"
        borderWidth="1px"
        borderColor={message.type === 'user' ? 'blue.100' : isArchitectResponse ? 'purple.100' : 'gray.100'}
        mb={5}
        transition="all 0.2s"
        _hover={{ boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)" }}
      >
        <VStack align="stretch" spacing={5} width="100%">
          <Box>
            <Text 
              fontSize="xs" 
              color={isArchitectResponse ? "purple.600" : "blue.600"} 
              fontWeight="600" 
              mb={1}
              textTransform="uppercase"
              letterSpacing="0.05em"
            >
              {isArchitectResponse ? "Data Architect" : message.type === 'user' ? "You" : "Assistant"}
            </Text>
            
            {isLoading ? (
              <VStack spacing={4} align="start" width="100%">
                <Text color="gray.600">Preparing detailed analysis...</Text>
                <Progress size="xs" isIndeterminate colorScheme="purple" width="100%" />
              </VStack>
            ) : (
              isArchitectResponse ? (
                <Box>
                  <FormattedMessage content={message.content} />
                </Box>
              ) : (
                <Text 
                  fontFamily="'Merriweather', Georgia, serif"
                  fontSize="16px"
                  fontWeight="500"
                  lineHeight="1.7"
                  color="gray.800"
                  whiteSpace="pre-wrap"
                >
                  {message.content}
                </Text>
              )
            )}
          </Box>

          {/* Feedback Section - Only show if feedback is required and not yet provided */}
          {requiresFeedback && (
            <Box 
              bg="gray.50" 
              p={4}
              borderRadius="md"
              borderWidth="1px"
              borderColor="gray.200"
              mt={2}
            >
              <Text 
                fontSize="15px" 
                color="gray.600" 
                fontWeight="500"
                mb={3}
                fontFamily="Georgia, serif"
              >
                Please review this understanding:
              </Text>
              <InlineFeedback 
                message={message}
                onFeedbackSubmit={handleFeedbackSubmit}
              />
            </Box>
          )}

          {/* Show feedback status badges */}
          {message.details?.feedback_status === 'approved' && !isArchitectResponse && (
            <Badge colorScheme="green" p={2} borderRadius="md">
              ✓ Understanding confirmed
            </Badge>
          )}
          
          {message.details?.feedback_status === 'needs_improvement' && (
            <Badge colorScheme="orange" p={2} borderRadius="md">
              Clarification requested: {message.details?.feedback_comments}
            </Badge>
          )}

          {/* Show pending status if no feedback given and in history tab */}
          {!message.details?.feedback_status && activeTab === 'history' && (
            <Badge colorScheme="blue" p={2} borderRadius="md">
              ⏳ Pending Review
            </Badge>
          )}

          {/* Analysis Details */}
          {message.details?.parsed_question && (
            <Accordion allowToggle width="100%">
              <AccordionItem border="none" borderTop="1px solid" borderColor="gray.200">
                <AccordionButton py={3} _hover={{ bg: "gray.50" }}>
                  <Box flex="1" textAlign="left">
                    <Text fontSize="15px" color="blue.600" fontWeight="500">
                      View Analysis Details
                    </Text>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} pt={2}>
                  <VStack align="stretch" spacing={4}>
                    <Box>
                      <Text fontWeight="bold" mb={2} color="gray.700">Business Analysis:</Text>
                      <VStack align="start" spacing={2} pl={4}>
                        <Text>Rephrased Question: {message.details.parsed_question.rephrased_question}</Text>
                        {message.details.parsed_question.business_context && (
                          <>
                            <Text>Domain: {message.details.parsed_question.business_context.domain}</Text>
                            <Text>Primary Objective: {message.details.parsed_question.business_context.primary_objective}</Text>
                            <Text>Key Entities:</Text>
                            <UnorderedList>
                              {message.details.parsed_question.business_context.key_entities?.map((entity, idx) => (
                                <ListItem key={idx}>{entity}</ListItem>
                              ))}
                            </UnorderedList>
                          </>
                        )}
                      </VStack>
                    </Box>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}

          {/* Data Architect Details - Show for architect responses */}
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
        </VStack>
      </Box>
    );
  };

  // Render conversation history sidebar
  const renderConversationHistory = () => (
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
  );

  // Update the useEffect hook to prevent auto-approval when switching tabs
  useEffect(() => {
    if (activeTab === 'chat') {
      // Clear messages and reset conversation when switching to chat tab
      setMessages([]);
      setActiveConversationId(null);
      setCurrentConversationId(null);
      setCurrentThreadId(null);
      
      // Update URL to /chat without thread ID
      navigate('/chat', { replace: true });
    } else if (activeTab === 'history') {
      // Don't auto-approve, just refresh the conversation list
      fetchConversations();
    }
  }, [activeTab, navigate]);

  // Add clearChat function
  const clearChat = () => {
    // Save current conversation ID before clearing
    const currentId = activeConversationId;
    
    // Clear messages and conversation ID
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    setCurrentThreadId(null);
    
    // Refresh conversations to ensure it shows in history
    fetchConversations();
    
    toast({
      title: 'Chat Cleared',
      description: 'The chat has been cleared. You can find it in the history tab.',
      status: 'info',
      duration: 3000
    });
  };

  // Also, let's add a useEffect to debug message changes
  useEffect(() => {
    console.log("Messages state changed:", messages);
    // Scroll to bottom whenever messages change
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Add this useEffect to monitor message state changes
  useEffect(() => {
    console.log("Messages state updated:", messages);
    
    // Check if we have an architect response
    const hasArchitectResponse = messages.some(
      msg => msg.details?.is_architect_response && !msg.details?.is_loading
    );
    
    if (hasArchitectResponse) {
      console.log("Architect response found in messages");
    }
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
            {renderConversationHistory()}
          </DrawerBody>
        </DrawerContent>
      </Drawer>
      
      <Flex h="100%" flexDirection="column">
        {/* Tab Navigation */}
        <Tabs 
          isFitted 
          variant="enclosed" 
          colorScheme="blue" 
          onChange={(index) => {
            const newTab = index === 0 ? 'chat' : 'history';
            setActiveTab(newTab);
            
            // If switching to chat tab, navigate to /chat
            if (newTab === 'chat') {
              navigate('/chat', { replace: true });
            }
          }}
          defaultIndex={activeTab === 'chat' ? 0 : 1}
        >
          <TabList mb="1em">
            <Tab>Chat</Tab>
            <Tab>History</Tab>
          </TabList>
          
          <TabPanels height="calc(100vh - 80px)">
            {/* Chat Tab */}
            <TabPanel p={0} height="100%">
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
                    <Heading size="md">New Chat</Heading>
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
                    {messages.map((message, index) => renderMessage({ ...message, id: index }))}
                    
                    {analysisResult && renderAnalysisConfirmation()}
                    
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
            </TabPanel>
            
            {/* History Tab */}
            <TabPanel p={0} height="100%">
              <Flex h="100%" direction={{ base: "column", lg: "row" }}>
                {/* Conversation List */}
                <Box 
                  width={{ base: "100%", lg: "350px" }} 
                  p={4} 
                  borderRightWidth={{ base: 0, lg: "1px" }}
                  borderBottomWidth={{ base: "1px", lg: 0 }}
                  borderColor="gray.200"
                  overflowY="auto"
                  height={{ base: "auto", lg: "100%" }}
                >
                  {isHistoryLoading ? (
                    <Progress size="xs" isIndeterminate colorScheme="blue" />
                  ) : conversations.length === 0 ? (
                    <Text color="gray.500" textAlign="center" py={4}>No conversation history</Text>
                  ) : (
                    <VStack spacing={3} align="stretch">
                      {conversations.map((conv) => (
                        <Box 
                          key={conv.id}
                          p={3}
                          borderRadius="md"
                          bg={activeConversationId === conv.id ? "blue.50" : "white"}
                          borderWidth="1px"
                          borderColor={activeConversationId === conv.id ? "blue.200" : "gray.200"}
                          cursor="pointer"
                          onClick={() => fetchConversation(conv.id)}
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
                </Box>
                
                {/* Selected Conversation Content */}
                <Box 
                  flex="1" 
                  p={4} 
                  overflowY="auto"
                  height="100%"
                  bg="gray.50"
                >
                  {activeConversationId ? (
                    <VStack spacing={6} align="stretch">
                      {messages.map((message, index) => renderMessage({ ...message, id: index }))}
                      <div ref={messagesEndRef} />
                    </VStack>
                  ) : (
                    <Flex 
                      height="100%" 
                      align="center" 
                      justify="center" 
                      direction="column"
                      color="gray.500"
                    >
                      <Box as={IoDocumentText} size="50px" mb={4} />
                      <Text fontSize="lg">Select a conversation from the list to view</Text>
                    </Flex>
                  )}
                </Box>
              </Flex>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Flex>
    </Box>
  )
}

export default ChatPage 