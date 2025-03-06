import React, { useState, useEffect } from 'react'
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

// Update the FormattedMessage component to properly handle Business Context
const FormattedMessage = ({ content }) => {
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
  
  // Format the main content
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
  
  // Format bullet points for Business Context
  const formatBusinessContext = () => {
    if (!businessContent) return null;
    
    // Check if content has bullet points
    const hasBullets = businessContent.includes('•');
    
    return (
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
        
        <VStack align="start" spacing={2} mt={3} pl={2}>
          {hasBullets ? (
            // Format as bullet points
            businessContent.split('•').filter(Boolean).map((bullet, bulletIdx) => (
              <HStack 
                key={bulletIdx} 
                spacing={3} 
                pl={2} 
                py={1}
                align="start"
                width="100%"
              >
                <Box 
                  w="6px" 
                  h="6px" 
                  bg="blue.500" 
                  borderRadius="full" 
                  mt={3}
                  flexShrink={0}
                />
                <Text 
                  color="gray.700"
                  fontSize="16px"
                  lineHeight="1.7"
                  fontFamily="'Merriweather', Georgia, serif"
                  fontWeight="400"
                >
                  {bullet.trim()}
                </Text>
              </HStack>
            ))
          ) : (
            // Format as regular text
            <Text 
              color="gray.700"
              fontSize="16px"
              lineHeight="1.7"
              fontFamily="'Merriweather', Georgia, serif"
              fontWeight="400"
              pl={2}
            >
              {businessContent}
            </Text>
          )}
        </VStack>
      </Box>
    );
  };
  
  // Format section content for accordions with modern table display
  const formatSectionContent = (content) => {
    // Check if this is the Available Tables section
    if (content.includes('LINEITEM:') || content.includes('ORDERS:') || content.includes('PART:') || 
        content.includes('SUPPLIER:') || content.includes('PARTSUPP:') || content.includes('NATION:')) {
      
      // This is a tables and columns section, format it as a structured display
      const tables = {};
      let currentTable = null;
      
      // Parse the content into a structured format
      content.split('\n').forEach(line => {
        const trimmedLine = line.trim();
        
        // Check if this is a table header (ends with a colon)
        if (trimmedLine.endsWith(':') && !trimmedLine.startsWith('•')) {
          currentTable = trimmedLine.replace(':', '');
          tables[currentTable] = [];
        } 
        // If it's a column definition (starts with spaces or tabs)
        else if (currentTable && trimmedLine.length > 0) {
          // Extract column name and description
          const columnMatch = trimmedLine.match(/([A-Z_]+)\s*-\s*(.*)/);
          if (columnMatch) {
            tables[currentTable].push({
              name: columnMatch[1],
              description: columnMatch[2]
            });
          } else {
            // Just add as a description line if it doesn't match the pattern
            tables[currentTable].push({
              name: '',
              description: trimmedLine
            });
          }
        }
      });
      
      // Check if we actually parsed any tables
      if (Object.keys(tables).length === 0) {
        // Fallback to regular formatting
        return content.split('\n').map((line, idx) => (
          <Text 
            key={idx} 
            color="gray.700"
            fontSize="16px"
            lineHeight="1.7"
            py={1}
            pl={2}
            fontFamily="'Merriweather', Georgia, serif"
          >
            {line.trim()}
          </Text>
        ));
      }
      
      // Render the structured tables with a modern design
      return (
        <VStack align="stretch" spacing={8} width="100%" mt={4}>
          {Object.keys(tables).map((tableName, tableIdx) => (
            <Box 
              key={tableIdx} 
              borderRadius="xl"
              overflow="hidden"
              boxShadow="0 4px 12px rgba(0, 0, 0, 0.08)"
              bg="white"
              position="relative"
            >
              {/* Table Header - Modern Design */}
              <Box 
                bg="blue.600" 
                color="white"
                p={4} 
                position="relative"
                overflow="hidden"
              >
                <Box 
                  position="absolute" 
                  top="0" 
                  right="0" 
                  bottom="0" 
                  left="0" 
                  bg="blue.500" 
                  opacity="0.3"
                  transform="skewX(-15deg) translateX(-10%)"
                />
                <Heading 
                  size="md" 
                  fontFamily="'Playfair Display', Georgia, serif"
                  color="white"
                  position="relative"
                  zIndex="1"
                >
                  {tableName}
                </Heading>
                <Text 
                  fontSize="sm" 
                  color="blue.100" 
                  mt={1}
                  position="relative"
                  zIndex="1"
                >
                  Database Table Schema
                </Text>
              </Box>
              
              {/* Columns - Modern Design */}
              <Box>
                {tables[tableName].map((column, colIdx) => (
                  <Box 
                    key={colIdx} 
                    p={4} 
                    borderBottomWidth={colIdx < tables[tableName].length - 1 ? "1px" : "0"}
                    borderColor="gray.100"
                    transition="all 0.2s"
                    _hover={{ bg: "gray.50" }}
                    display="flex"
                    flexDirection={["column", "row"]}
                    alignItems={["flex-start", "center"]}
                  >
                    {column.name ? (
                      <>
                        <Box 
                          width={["100%", "30%"]} 
                          mb={[2, 0]}
                          pr={4}
                        >
                          <HStack spacing={2} align="center">
                            <Box 
                              w="8px" 
                              h="8px" 
                              bg="blue.500" 
                              borderRadius="full" 
                            />
                            <Text 
                              fontFamily="'Courier New', monospace" 
                              fontWeight="600" 
                              color="blue.700"
                              fontSize="16px"
                              letterSpacing="0.02em"
                            >
                              {column.name}
                            </Text>
                          </HStack>
                        </Box>
                        <Box width={["100%", "70%"]}>
                          <Text 
                            fontFamily="'Merriweather', Georgia, serif"
                            fontSize="15px"
                            color="gray.700"
                            lineHeight="1.6"
                          >
                            {column.description}
                          </Text>
                        </Box>
                      </>
                    ) : (
                      <Box width="100%">
                        <Text 
                          fontFamily="'Merriweather', Georgia, serif"
                          fontSize="15px"
                          color="gray.600"
                          fontStyle="italic"
                        >
                          {column.description}
                        </Text>
                      </Box>
                    )}
                  </Box>
                ))}
              </Box>
            </Box>
          ))}
        </VStack>
      );
    } else {
      // For non-table content, use the existing formatting
      return content.split('\n').map((line, idx) => {
        if (line.trim().startsWith('•')) {
          return (
            <HStack 
              key={idx} 
              spacing={3} 
              pl={2} 
              py={1}
              align="start"
            >
              <Box 
                w="6px" 
                h="6px" 
                bg="blue.500" 
                borderRadius="full" 
                mt={3}
                flexShrink={0}
              />
              <Text 
                color="gray.700"
                fontSize="16px"
                lineHeight="1.7"
                fontFamily="'Merriweather', Georgia, serif"
                fontWeight="400"
              >
                {line.replace('•', '').trim()}
              </Text>
            </HStack>
          );
        } else {
          return (
            <Text 
              key={idx} 
              color="gray.700"
              fontSize="16px"
              lineHeight="1.7"
              py={1}
              pl={2}
              fontFamily="'Merriweather', Georgia, serif"
              letterSpacing="0.01em"
            >
              {line.trim()}
            </Text>
          );
        }
      });
    }
  };
  
  return (
    <VStack align="start" spacing={4} width="100%">
      {/* Main content */}
      {formattedMainContent}
      
      {/* Business Context section */}
      {hasBusinessContext && formatBusinessContext()}
      
      {/* Implementation Details accordion */}
      {hasImplementationDetails && (
        <Accordion allowToggle width="100%" mt={3}>
          <AccordionItem border="none" borderTop="1px solid" borderColor="gray.200">
            <AccordionButton 
              px={0} 
              py={3}
              _hover={{ bg: 'transparent', color: 'blue.500' }}
            >
              <Heading 
                size="sm" 
                color="blue.600"
                fontWeight="600"
                fontFamily="'Playfair Display', Georgia, serif"
                fontSize="16px"
                flex="1"
                textAlign="left"
              >
                Implementation Details
              </Heading>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4} pl={3}>
              {formatSectionContent(implementationContent)}
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      )}
      
      {/* Available Tables accordion */}
      {hasAvailableTables && (
        <Accordion allowToggle width="100%" mt={2}>
          <AccordionItem border="none" borderTop="1px solid" borderColor="gray.200">
            <AccordionButton 
              px={0} 
              py={3}
              _hover={{ bg: 'transparent', color: 'blue.500' }}
            >
              <Heading 
                size="sm" 
                color="blue.600"
                fontWeight="600"
                fontFamily="'Playfair Display', Georgia, serif"
                fontSize="16px"
                flex="1"
                textAlign="left"
              >
                Available Tables and Columns
              </Heading>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4} pl={3}>
              {formatSectionContent(tablesContent)}
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      )}
    </VStack>
  );
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

  const handleSubmit = async (isApproved) => {
    try {
      setIsSubmitting(true);
      await onFeedbackSubmit({
        feedback_id: message.details.feedback_id,
        conversation_id: message.details.conversation_id,
        approved: isApproved,
        comments: comments,
        suggested_changes: null
      });
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const confidence = message.details.parsed_question.confidence_score || 0;
  const confidenceColor = confidence > 0.8 ? "green" : confidence > 0.5 ? "orange" : "red";

  return (
    <VStack align="stretch" spacing={4} bg="gray.50" p={4} borderRadius="lg">
      <HStack justify="space-between" align="center">
        <Text fontSize="lg" fontWeight="medium" color="blue.700">
          Business Analysis Review
        </Text>
        <Tag colorScheme={confidenceColor}>
          Confidence: {Math.round(confidence * 100)}%
        </Tag>
      </HStack>

      {/* Business Understanding */}
      <Box bg="white" p={4} borderRadius="md" shadow="sm">
        <Text fontWeight="medium" mb={2}>
          Business Understanding:
        </Text>
        <Text color="gray.700">
          {message.details.parsed_question.rephrased_question}
        </Text>
      </Box>

      {/* Business Context */}
      <Box bg="white" p={4} borderRadius="md" shadow="sm">
        <Text fontWeight="medium" mb={2}>
          Business Context:
        </Text>
        <VStack align="start" spacing={2}>
          <Text><strong>Domain:</strong> {message.details.parsed_question.business_context.domain}</Text>
          <Text><strong>Objective:</strong> {message.details.parsed_question.business_context.primary_objective}</Text>
          <Text><strong>Key Entities:</strong> {message.details.parsed_question.business_context.key_entities.join(', ')}</Text>
          <Text><strong>Impact:</strong> {message.details.parsed_question.business_context.business_impact}</Text>
        </VStack>
      </Box>

      {/* Key Points */}
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

      {/* Assumptions */}
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

      {/* Clarifying Questions */}
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
  const toast = useToast();

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
          message: userMessage
        })
      })

      const data = await response.json()
      console.log("Response from server:", data)

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
  const renderMessage = (message) => (
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

          {/* Show suggested questions with improved styling */}
          {message.details?.suggested_questions && (
            <VStack align="stretch" mt={3} spacing={3}>
              <Text 
                fontWeight="600" 
                color="blue.700" 
                fontSize="15px"
                borderBottom="2px solid"
                borderColor="blue.100"
                pb={1}
                width="fit-content"
              >
                Related Questions
              </Text>
              <SimpleGrid columns={[1, null, 2]} spacing={3}>
                {message.details.suggested_questions.map((question, idx) => (
                  <Button
                    key={idx}
                    variant="outline"
                    size="md"
                    colorScheme="blue"
                    leftIcon={<IoAdd />}
                    onClick={() => setInput(question)}
                    justifyContent="flex-start"
                    whiteSpace="normal"
                    textAlign="left"
                    height="auto"
                    py={3}
                    px={4}
                    borderRadius="md"
                    fontWeight="500"
                    fontSize="15px"
                    transition="all 0.2s"
                    _hover={{
                      bg: "blue.50",
                      transform: "translateY(-2px)",
                      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)"
                    }}
                  >
                    {question}
                  </Button>
                ))}
              </SimpleGrid>
            </VStack>
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
                  {message.details?.analysis && (
                    <Box>
                      <Text fontWeight="bold" mb={2} color="gray.700">Business Analysis:</Text>
                      <VStack align="start" spacing={2} pl={4}>
                        <Text>Primary Objective: {message.details.analysis.business_context?.primary_objective}</Text>
                        <Text>Domain: {message.details.analysis.business_context?.domain}</Text>
                        <Text>Key Entities:</Text>
                        <UnorderedList>
                          {message.details.analysis.business_context?.key_entities?.map((entity, idx) => (
                            <ListItem key={idx}>{entity}</ListItem>
                          ))}
                        </UnorderedList>
                      </VStack>
                    </Box>
                  )}
                  {message.details?.sources && (
                    <Box>
                      <Text fontWeight="bold" mb={2} color="gray.700">Source Documents:</Text>
                      <VStack align="start" spacing={3} pl={4}>
                        {message.details.sources.doc_results?.map((doc, idx) => (
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

  return (
    <Box p={6} maxW="1400px" mx="auto">
      <VStack spacing={6} mb={8} align="stretch">
        {messages.map((message, index) => renderMessage({ ...message, id: index }))}
        
        {analysisResult && renderAnalysisConfirmation()}
        
        {processingStep && (
          <Box 
            p={4} 
            bg="blue.50" 
            borderRadius="md" 
            width={["98%", "95%", "90%"]}  // Increased width to match messages
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
      </VStack>

      <Box 
        maxW="1200px" 
        mx="auto" 
        p={4} 
        borderRadius="lg" 
        borderWidth="1px" 
        borderColor="gray.200"
        bg="white"
        boxShadow="0 2px 10px rgba(0, 0, 0, 0.05)"
        width={["98%", "95%", "90%"]}  // Match the width of messages
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
  )
}

export default ChatPage 