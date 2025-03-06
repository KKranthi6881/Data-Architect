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

// Update the FormattedMessage component to handle different content types
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
  
  // Check if content contains sections we want to hide in accordions
  const hasImplementationDetails = content.includes('**Implementation Details:**');
  const hasAvailableTables = content.includes('**Available Tables and Columns:**');
  const hasBusinessContext = content.includes('**Business Context:**');
  
  // Function to extract section content
  const extractSectionContent = (sectionTitle) => {
    const startMarker = `**${sectionTitle}:**`;
    const startIndex = content.indexOf(startMarker);
    if (startIndex === -1) return '';
    
    // Find the next section marker after this one
    const nextSectionIndex = content.indexOf('**', startIndex + startMarker.length);
    
    if (nextSectionIndex === -1) {
      // This is the last section
      return content.substring(startIndex + startMarker.length).trim();
    } else {
      // Extract content until the next section
      return content.substring(startIndex + startMarker.length, nextSectionIndex).trim();
    }
  };
  
  // Extract content for sections
  const implementationContent = hasImplementationDetails ? 
    extractSectionContent('Implementation Details') : '';
  
  const tablesContent = hasAvailableTables ? 
    extractSectionContent('Available Tables and Columns') : '';
    
  const businessContent = hasBusinessContext ?
    extractSectionContent('Business Context') : '';
  
  // Remove these sections from the main content
  let mainContent = content;
  
  if (hasBusinessContext) {
    const startMarker = '**Business Context:**';
    const startIndex = mainContent.indexOf(startMarker);
    const nextSectionIndex = mainContent.indexOf('**', startIndex + startMarker.length);
    
    if (nextSectionIndex === -1) {
      // This is the last section
      mainContent = mainContent.substring(0, startIndex);
    } else {
      // Remove just this section
      mainContent = mainContent.substring(0, startIndex) + 
                   mainContent.substring(nextSectionIndex);
    }
  }
  
  if (hasImplementationDetails) {
    const startMarker = '**Implementation Details:**';
    const startIndex = mainContent.indexOf(startMarker);
    const nextSectionIndex = mainContent.indexOf('**', startIndex + startMarker.length);
    
    if (nextSectionIndex === -1) {
      // This is the last section
      mainContent = mainContent.substring(0, startIndex);
    } else {
      // Remove just this section
      mainContent = mainContent.substring(0, startIndex) + 
                   mainContent.substring(nextSectionIndex);
    }
  }
  
  if (hasAvailableTables) {
    const startMarker = '**Available Tables and Columns:**';
    const startIndex = mainContent.indexOf(startMarker);
    const nextSectionIndex = mainContent.indexOf('**', startIndex + startMarker.length);
    
    if (nextSectionIndex === -1) {
      // This is the last section
      mainContent = mainContent.substring(0, startIndex);
    } else {
      // Remove just this section
      mainContent = mainContent.substring(0, startIndex) + 
                   mainContent.substring(nextSectionIndex);
    }
  }
  
  // Check if the content has markdown formatting
  const hasMarkdown = mainContent.includes('**') || mainContent.includes('##');
  
  if (hasMarkdown) {
    // Format the main content with markdown
    const formattedMainContent = mainContent.split('**').map((part, idx) => {
      if (idx % 2 === 0) {
        // Regular text
        return (
          <Text 
            key={idx} 
            fontSize="16px"
            fontFamily="'Merriweather', Georgia, serif"
            lineHeight="1.7"
            color="gray.800"
            mb={3}
            letterSpacing="0.01em"
          >
            {part}
          </Text>
        );
      } else {
        // This is a section title
        const isUnderstanding = part.includes('Based on your question and available content, I assume:');
        
        return (
          <Box key={idx} width="100%" mt={5} mb={4}>
            <Heading 
              size="md" 
              color={isUnderstanding ? "purple.700" : "gray.800"}
              fontWeight="600"
              fontFamily="'Playfair Display', Georgia, serif"
              pb={2}
              borderBottom="2px solid"
              borderColor={isUnderstanding ? "purple.200" : "gray.200"}
              width="fit-content"
              fontSize={isUnderstanding ? "20px" : "18px"}
              letterSpacing="0.02em"
            >
              {part}
            </Heading>
          </Box>
        );
      }
    });
    
    return (
      <>
        {formattedMainContent}
        
        {/* Business Context Section if available */}
        {businessContent && (
          <Box width="100%" mt={5} mb={4}>
            <Heading 
              size="md" 
              color="blue.700"
              fontWeight="600"
              fontFamily="'Playfair Display', Georgia, serif"
              pb={2}
              borderBottom="2px solid"
              borderColor="blue.200"
              width="fit-content"
              fontSize="18px"
              letterSpacing="0.02em"
            >
              Business Context
            </Heading>
            
            <Text 
              mt={3}
              fontSize="16px"
              fontFamily="'Merriweather', Georgia, serif"
              lineHeight="1.7"
              color="gray.800"
            >
              {businessContent}
            </Text>
          </Box>
        )}
      </>
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

const InlineFeedback = ({ message, onFeedbackSubmit }) => {
  const [comments, setComments] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (approved) => {
    if (isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      await onFeedbackSubmit({
        feedback_id: message.details?.feedback_id,
        conversation_id: message.details?.conversation_id,
        approved,
        comments: comments
      });
      
      // Clear comments after submission
      setComments('');
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <VStack align="stretch" spacing={3}>
      {/* Key Points */}
      {message.details?.parsed_question?.key_points && (
        <Box bg="white" p={4} borderRadius="md" shadow="sm">
          <Text fontWeight="medium" mb={2}>
            Key Business Points:
          </Text>
          <UnorderedList spacing={1}>
            {message.details.parsed_question.key_points.map((point, idx) => (
              <ListItem key={idx}>{point}</ListItem>
            ))}
          </UnorderedList>
        </Box>
      )}

      {/* Assumptions */}
      {message.details?.parsed_question?.assumptions && (
        <Box bg="white" p={4} borderRadius="md" shadow="sm">
          <Text fontWeight="medium" mb={2}>
            Assumptions to Verify:
          </Text>
          <UnorderedList spacing={1}>
            {message.details.parsed_question.assumptions.map((assumption, idx) => (
              <ListItem key={idx}>{assumption}</ListItem>
            ))}
          </UnorderedList>
        </Box>
      )}

      {/* Clarifying Questions */}
      {message.details?.parsed_question?.clarifying_questions && (
        <Box bg="white" p={4} borderRadius="md" shadow="sm">
          <Text fontWeight="medium" mb={2}>
            Clarifying Questions:
          </Text>
          <UnorderedList spacing={1}>
            {message.details.parsed_question.clarifying_questions.map((question, idx) => (
              <ListItem key={idx}>{question}</ListItem>
            ))}
          </UnorderedList>
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
        
        // Check if we have the required fields
        if (!conversation.query) {
          console.warn("Missing query in conversation data");
        }
        if (!conversation.response) {
          console.warn("Missing response in conversation data");
        }
        
        // Clear existing messages
        setMessages([]);
        
        // Add user message
        const userMessage = {
          type: 'user',
          content: conversation.query || "No query available",
          id: `${id}-query`
        };
        console.log("Adding user message:", userMessage);
        setMessages(prev => [...prev, userMessage]);
        
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
        setMessages(prev => [...prev, assistantMessage]);
        
        setActiveConversationId(id);
        
        // Update URL without reloading
        navigate(`/chat/${id}`, { replace: true });
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
    if (!input.trim()) return
    
    setLoading(true)
    const userMessage = input
    setInput('')
    
    // Add user message
    setMessages(prev => [...prev, { 
      type: 'user', 
      content: userMessage 
    }])

    try {
      // Show processing steps
      setProcessingStep('Analyzing code and documentation...')
      
      const response = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: activeConversationId // Include existing conversation ID if available
        })
      })

      const data = await response.json()
      console.log("Response from server:", data)

      // Set the active conversation ID from the response
      setActiveConversationId(data.conversation_id);
      
      // Update URL without reloading
      navigate(`/chat/${data.conversation_id}`, { replace: true });

      // Add assistant response
      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        details: {
          ...data,
          conversation_id: data.conversation_id,
          feedback_id: data.feedback_id,
          feedback_required: data.feedback_required,
          feedback_status: 'pending'
        }
      }
      
      setMessages(prev => [...prev, assistantMessage])
      
      // Refresh conversation list
      fetchConversations();

    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Error processing your request'
      }])
    }

    setProcessingStep('')
    setLoading(false)
  }

  const handleFeedbackSubmit = async (feedback) => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/feedback/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to submit feedback');
      }

      const data = await response.json();
      
      if (feedback.approved) {
        // Update the message to show it's approved
        setMessages(prev => prev.map(msg => 
          msg.details?.feedback_id === feedback.feedback_id
            ? {
                ...msg,
                content: msg.content.replace('_Waiting for human feedback to ensure accuracy..._', '_Response approved by human reviewer_'),
                details: {
                  ...msg.details,
                  feedback_status: 'approved'
                }
              }
            : msg
        ));
        
        toast({
          title: 'Response Approved',
          description: 'The response has been approved and finalized.',
          status: 'success',
          duration: 3000
        });
      } else {
        // If not approved, send a follow-up request with feedback
        const followUpResponse = await fetch('http://localhost:8000/chat/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: feedback.feedback_id,
            feedback: {
              approved: false,
              comments: feedback.comments,
              feedback_id: feedback.feedback_id,
              request_type: 'feedback_improvement'
            }
          })
        });

        if (!followUpResponse.ok) {
          throw new Error('Failed to get improved response');
        }

        const followUpData = await followUpResponse.json();
        
        // Add the new response with feedback context
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: followUpData.answer,
          details: {
            ...followUpData,
            feedback_status: 'pending',
            previous_feedback: feedback.comments
          }
        }]);
        
        // Update the original message status
        setMessages(prev => prev.map(msg => 
          msg.details?.feedback_id === feedback.feedback_id
            ? {
                ...msg,
                details: {
                  ...msg.details,
                  feedback_status: 'rejected',
                  feedback_comments: feedback.comments
                }
              }
            : msg
        ));
        
        toast({
          title: 'Feedback Submitted',
          description: 'Generating improved response based on your feedback...',
          status: 'info',
          duration: 3000
        });
      }
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  // Update message display with wider boxes
  const renderMessage = (message) => {
    console.log("Rendering message:", message); // Add this for debugging
    
    return (
      <Box 
        key={message.id}
        bg={message.type === 'user' ? 'blue.50' : 'white'}
        p={5}
        borderRadius="lg"
        alignSelf="flex-start"  // Always align to the left
        width={["98%", "95%", "90%"]}  // Increased width for all messages
        boxShadow="0 2px 8px rgba(0, 0, 0, 0.08)"
        borderWidth="1px"
        borderColor={message.type === 'user' ? 'blue.100' : 'gray.100'}
        mb={5}
        transition="all 0.2s"
        _hover={{ boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)" }}
      >
        {message.type === 'user' ? (
          <VStack align="stretch" spacing={3} width="100%">
            <Box>
              <Text 
                fontSize="xs" 
                color="blue.600" 
                fontWeight="600" 
                mb={1}
                textTransform="uppercase"
                letterSpacing="0.05em"
              >
                You
              </Text>
              <Text 
                fontFamily="'Merriweather', Georgia, serif"
                fontSize="16px"
                fontWeight="500"
                lineHeight="1.7"
                color="gray.800"
                whiteSpace="pre-wrap"  // Preserve line breaks in user messages
              >
                {message.content}
              </Text>
            </Box>
          </VStack>
        ) : (
          <VStack align="stretch" spacing={5} width="100%">
            <Box>
              <Text 
                fontSize="xs" 
                color="purple.600" 
                fontWeight="600" 
                mb={1}
                textTransform="uppercase"
                letterSpacing="0.05em"
              >
                Assistant
              </Text>
              {/* Main content with formatting */}
              <FormattedMessage content={message.content} />
            </Box>

            {/* Feedback Section */}
            {message.details?.feedback_status === 'pending' && (
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
                  Waiting for review...
                </Text>
                <InlineFeedback 
                  message={message}
                  onFeedbackSubmit={handleFeedbackSubmit}
                />
              </Box>
            )}

            {/* Analysis Details with improved styling */}
            <Accordion allowToggle>
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
                    {message.details?.parsed_question && (
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
                    )}
                    {message.details?.sources && (
                      <Box>
                        <Text fontWeight="bold" mb={2} color="gray.700">Source Documents:</Text>
                        <VStack align="start" spacing={3} pl={4}>
                          {message.details.sources.results?.map((doc, idx) => (
                            <Box key={idx} p={3} bg="gray.50" borderRadius="md" w="100%">
                              <Text fontSize="15px">{doc.content}</Text>
                            </Box>
                          ))}
                        </VStack>
                      </Box>
                    )}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </VStack>
        )}
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
          {conversations.map((conv) => (
            <Box 
              key={conv.id}
              p={3}
              borderRadius="md"
              bg={activeConversationId === conv.id ? "blue.50" : "white"}
              borderWidth="1px"
              borderColor={activeConversationId === conv.id ? "blue.200" : "gray.200"}
              cursor="pointer"
              onClick={() => handleConversationSelect(conv.id)}
              _hover={{ bg: activeConversationId === conv.id ? "blue.50" : "gray.50" }}
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
                <Badge 
                  colorScheme={
                    conv.feedback_status === 'approved' ? 'green' : 
                    conv.feedback_status === 'needs_improvement' ? 'orange' : 'blue'
                  }
                  fontSize="xs"
                >
                  {
                    conv.feedback_status === 'approved' ? 'Approved' : 
                    conv.feedback_status === 'needs_improvement' ? 'Needs Review' : 'Pending'
                  }
                </Badge>
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
      
      <Flex h="100%">
        {/* Desktop sidebar */}
        <Box 
          w="300px" 
          h="100%" 
          borderRightWidth="1px" 
          borderColor="gray.200"
          display={{ base: "none", md: "block" }}
          overflowY="auto"
        >
          {renderConversationHistory()}
        </Box>
        
        {/* Main chat area */}
        <Box flex="1" h="100%" display="flex" flexDirection="column">
          {/* Header with mobile menu */}
          <Flex 
            p={4} 
            borderBottomWidth="1px" 
            borderColor="gray.200" 
            align="center"
            bg="white"
          >
            <IconButton
              icon={<IoMenu />}
              aria-label="Open menu"
              display={{ base: "flex", md: "none" }}
              mr={3}
              onClick={onOpen}
            />
            <Heading size="md">
              {activeConversationId ? 'Conversation' : 'New Chat'}
            </Heading>
            {activeConversationId && (
              <Button 
                ml="auto" 
                leftIcon={<IoAdd />} 
                colorScheme="blue" 
                size="sm"
                onClick={startNewConversation}
              >
                New Chat
              </Button>
            )}
          </Flex>
          
          {/* Messages area */}
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
          
          {/* Input area */}
          <Box 
            p={4} 
            borderTopWidth="1px" 
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
        </Box>
      </Flex>
    </Box>
  )
}

export default ChatPage 