import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Container,
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
  useToast,
  Icon
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
  IoClose,
  IoRefresh,
  IoArrowForward,
  IoCopy, 
  IoCheckmarkDone,
  IoDownload,
  IoCode
} from 'react-icons/io5'
import { useParams, useNavigate } from 'react-router-dom'
import { CodeDisplay } from '../components/CodeDisplay'
import { Prism } from 'react-syntax-highlighter'
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import { v4 as uuidv4 } from 'uuid'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism'

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

const ChatMessage = ({ message }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Format the message content sections
  const formatMessageContent = () => {
    const details = message.details?.question_analysis;
    if (!details) return null;

    return (
      <VStack align="start" spacing={3} width="100%">
        {/* Business Understanding */}
        <Box>
          <Text fontWeight="medium">Business Understanding:</Text>
          <Text>{details.rephrased_question}</Text>
        </Box>

        {/* Key Points */}
        <Box>
          <Text fontWeight="medium">Key Business Points:</Text>
          <UnorderedList>
            {details.key_points?.map((point, idx) => (
              <ListItem key={idx}>{point}</ListItem>
            ))}
          </UnorderedList>
        </Box>
        
        {/* Technical Context */}
        <Box>
          <Text fontWeight="medium">Technical Analysis:</Text>
          <Text>{details.technical_context?.analysis || "No technical analysis available"}</Text>
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
      {message.role === 'assistant' && message.details && (
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
                <Text fontWeight="medium" color="blue.600">View Analysis Details</Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              {formatMessageContent()}
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      )}
    </VStack>
  );
};

// Add this component for section display
const ArchitectSection = ({ title, content }) => {
  // Extract code blocks if present
  const codeBlocks = content.match(/```(\w+)?\s*([\s\S]*?)```/g) || [];
  const textContent = content.replace(/```(\w+)?\s*([\s\S]*?)```/g, '').trim();

  return (
    <Box mb={6} borderLeft="4px" borderColor="purple.200" pl={4}>
      <Heading size="md" mb={3} color="purple.700">
        {title}
      </Heading>
      
      {textContent && (
        <VStack align="stretch" spacing={2} mb={codeBlocks.length > 0 ? 4 : 0}>
          {textContent.split('\n').map((line, i) => {
            if (line.trim().startsWith('-')) {
              return (
                <HStack key={i} align="start" spacing={2}>
                  <Box color="purple.500">•</Box>
                  <Text>{line.replace('-', '').trim()}</Text>
                </HStack>
              );
            }
            if (line.trim().startsWith('1.') || line.trim().startsWith('2.') || line.trim().startsWith('3.')) {
              return (
                <Text key={i} pl={4} fontWeight="medium">
                  {line.trim()}
                </Text>
              );
            }
            return <Text key={i}>{line}</Text>;
          })}
        </VStack>
      )}

      {codeBlocks.map((block, index) => {
        const [, lang, code] = block.match(/```(\w+)?\s*([\s\S]*?)```/) || [];
        return (
          <CodeDisplay
            key={index}
            code={code.trim()}
            language={lang?.toLowerCase() || 'sql'}
          />
        );
      })}
    </Box>
  );
};

// Update the architect response rendering in your message component
const renderArchitectResponse = (architectResponse) => {
  if (!architectResponse?.sections) return null;

  return (
    <Box 
      bg="purple.50" 
      p={6} 
      borderRadius="lg" 
      border="1px" 
      borderColor="purple.200"
      mt={4}
    >
      <Heading size="lg" mb={6} color="purple.800">
        Data Architect Analysis
      </Heading>

      {Object.entries(architectResponse.sections).map(([key, content]) => (
        <ArchitectSection 
          key={key} 
          title={key.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')} 
          content={content} 
        />
      ))}

      {architectResponse.implementation && (
        <Box mt={6}>
          <Heading size="md" mb={4} color="purple.700">
            Implementation Details
          </Heading>
          
          {architectResponse.implementation.sql?.map((sql, i) => (
            <CodeDisplay
              key={`sql-${i}`}
              code={sql}
              language="sql"
              fileName={`model_${i + 1}.sql`}
            />
          ))}
          
          {architectResponse.implementation.yaml?.map((yaml, i) => (
            <CodeDisplay
              key={`yaml-${i}`}
              code={yaml}
              language="yaml"
              fileName={`schema_${i + 1}.yml`}
            />
          ))}
        </Box>
      )}
    </Box>
  );
};

// Add this function in your component to format code blocks
const formatMessageContent = (content) => {
  if (!content) return '';
  
  // Check if content contains code blocks with triple backticks
  if (content.includes('```')) {
    const parts = [];
    let currentIndex = 0;
    let codeBlockStart = content.indexOf('```', currentIndex);
    
    // Process each code block
    while (codeBlockStart !== -1) {
      // Add text before code block
      if (codeBlockStart > currentIndex) {
        parts.push({
          type: 'text',
          content: content.substring(currentIndex, codeBlockStart)
        });
      }
      
      // Find the end of the code block
      const codeBlockEnd = content.indexOf('```', codeBlockStart + 3);
      if (codeBlockEnd === -1) {
        // No closing backticks, treat rest as text
        parts.push({
          type: 'text',
          content: content.substring(codeBlockStart)
        });
        break;
      }
      
      // Extract language and code
      const codeWithLang = content.substring(codeBlockStart + 3, codeBlockEnd);
      const firstLineBreak = codeWithLang.indexOf('\n');
      const language = firstLineBreak > 0 ? codeWithLang.substring(0, firstLineBreak).trim() : '';
      const code = firstLineBreak > 0 ? codeWithLang.substring(firstLineBreak + 1) : codeWithLang;
      
      parts.push({
        type: 'code',
        language: language || 'sql', // Default to SQL if no language specified
        content: code
      });
      
      currentIndex = codeBlockEnd + 3;
      codeBlockStart = content.indexOf('```', currentIndex);
    }
    
    // Add remaining text after last code block
    if (currentIndex < content.length) {
      parts.push({
        type: 'text',
        content: content.substring(currentIndex)
      });
    }
    
    return parts;
  }
  
  // If no code blocks, return as single text part
  return [{ type: 'text', content }];
};

// Add a CodeBlock component for better code display
const CodeBlock = ({ code, language }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `snippet.${language || 'txt'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  return (
    <Box
      position="relative"
      my={4}
      borderRadius="md"
      overflow="hidden"
      fontSize="sm"
    >
      <HStack
        bg="gray.700"
        color="white"
        p={2}
        justify="space-between"
        align="center"
      >
        <Badge colorScheme="blue" variant="solid">
          {language || 'code'}
        </Badge>
        <HStack>
          <IconButton
            icon={copied ? <IoCheckmarkDone /> : <IoCopy />}
            size="sm"
            variant="ghost"
            colorScheme={copied ? "green" : "gray"}
            onClick={handleCopy}
            aria-label="Copy code"
            title="Copy to clipboard"
          />
          <IconButton
            icon={<IoDownload />}
            size="sm"
            variant="ghost"
            onClick={handleDownload}
            aria-label="Download code"
            title="Download code"
          />
        </HStack>
      </HStack>
      <Prism
        language={language || 'sql'}
        style={atomDark}
        customStyle={{
          margin: 0,
          padding: '16px',
          maxHeight: '400px',
          overflow: 'auto'
        }}
      >
        {code}
      </Prism>
    </Box>
  );
};

// Add this function to render message content with proper formatting
const renderMessageContent = (content) => {
  const formattedContent = formatMessageContent(content);
  
  if (Array.isArray(formattedContent)) {
    return (
      <>
        {formattedContent.map((part, index) => {
          if (part.type === 'text') {
            return (
              <Text key={index} whiteSpace="pre-wrap">
                {part.content}
              </Text>
            );
          } else if (part.type === 'code') {
            return (
              <CodeBlock
                key={index}
                code={part.content}
                language={part.language}
              />
            );
          }
          return null;
        })}
      </>
    );
  }
  
  return <Text whiteSpace="pre-wrap">{content}</Text>;
};

// Add this component to better display code results

const CodeResultDisplay = ({ result }) => {
  if (!result) return null;
  
  const { file, repository, content } = result;
  
  return (
    <Box 
      borderWidth="1px" 
      borderRadius="md" 
      p={3} 
      mb={3}
      bg="gray.50"
    >
      <HStack mb={2}>
        <Text fontWeight="bold">{file?.path || 'Unknown file'}</Text>
        {repository && (
          <Badge colorScheme="purple">
            {repository.name}
          </Badge>
        )}
      </HStack>
      
      <Code p={2} borderRadius="md" fontSize="sm" overflowX="auto" whiteSpace="pre">
        {content || 'No content available'}
      </Code>
      
      {result.dbt_info && (
        <Box mt={2} p={2} bg="purple.50" borderRadius="md">
          <Text fontWeight="bold" mb={1}>DBT Model Info</Text>
          {result.dbt_info.model_name && (
            <Text fontSize="sm">Model: {result.dbt_info.model_name}</Text>
          )}
          {result.dbt_info.materialization && (
            <Text fontSize="sm">Materialization: {result.dbt_info.materialization}</Text>
          )}
          {result.dbt_info.references && result.dbt_info.references.length > 0 && (
            <Text fontSize="sm">References: {result.dbt_info.references.join(', ')}</Text>
          )}
        </Box>
      )}
    </Box>
  );
};

// Update the renderMessage function to better display architect responses

const renderMessage = (message) => {
  if (!message) return null;
  
  const isUser = message.role === 'user';
  const isArchitectResponse = message.role === 'assistant' && message.type === 'architect';
  
  return (
    <Box
      key={message.id}
      bg={isUser ? 'blue.50' : isArchitectResponse ? 'purple.50' : 'white'}
      p={4}
      borderRadius="md"
      maxWidth={isUser ? '70%' : '90%'}
      alignSelf={isUser ? 'flex-end' : 'flex-start'}
      boxShadow="sm"
      mb={4}
      border="1px solid"
      borderColor={isUser ? 'blue.100' : isArchitectResponse ? 'purple.100' : 'gray.200'}
    >
      <VStack align="stretch" spacing={3}>
        <HStack>
          <Avatar 
            size="sm" 
            bg={isUser ? 'blue.500' : isArchitectResponse ? 'purple.500' : 'green.500'} 
            name={isUser ? 'You' : isArchitectResponse ? 'Data Architect' : 'Assistant'} 
          />
          <Text fontWeight="bold" color={
            isUser ? 'blue.700' : 
            isArchitectResponse ? 'purple.700' : 
            'green.700'
          }>
            {isUser ? 'You' : isArchitectResponse ? 'Data Architect' : 'Assistant'}
          </Text>
          
          {isArchitectResponse && message.details?.processing_time && (
            <Badge colorScheme="purple" ml={2}>
              {message.details.processing_time.toFixed(1)}s
            </Badge>
          )}
          
          {isArchitectResponse && message.details?.question_type && (
            <Badge colorScheme="blue" ml={2}>
              {message.details.question_type}
            </Badge>
          )}
        </HStack>
        
        <Box flex="1">
          {renderMessageContent(message.content)}
        </Box>
        
        {/* Show GitHub results */}
        {isArchitectResponse && message.details?.github_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      GitHub Code Results ({message.details.github_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.github_results.map((result, idx) => (
                      <CodeResultDisplay key={idx} result={result} />
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
        
        {/* Show SQL results */}
        {isArchitectResponse && message.details?.sql_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      SQL Schema Results ({message.details.sql_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.sql_results.map((result, idx) => (
                      <Box 
                        key={idx}
                        p={3}
                        bg="gray.50"
                        borderRadius="md"
                        borderLeft="3px solid"
                        borderColor="blue.300"
                      >
                        <Text fontWeight="medium">{result.metadata?.source || 'SQL Schema'}</Text>
                        <Code p={2} mt={2} fontSize="sm" overflowX="auto" whiteSpace="pre">
                          {result.content || 'No SQL content available'}
                        </Code>
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
        
        {/* Show DBT results */}
        {isArchitectResponse && message.details?.dbt_results?.length > 0 && (
          <Box mt={3}>
            <Accordion allowToggle>
              <AccordionItem>
                <h2>
                  <AccordionButton>
                    <Box flex="1" textAlign="left" fontWeight="medium">
                      DBT Model Results ({message.details.dbt_results.length})
                    </Box>
                    <AccordionIcon />
                  </AccordionButton>
                </h2>
                <AccordionPanel pb={4}>
                  <VStack align="stretch" spacing={2}>
                    {message.details.dbt_results.map((result, idx) => (
                      <Box key={idx} p={4} borderWidth="1px" borderRadius="md">
                        <Text fontWeight="medium">{result.model_name}</Text>
                        <Text fontSize="sm" color="gray.500">{result.description}</Text>
                        {result.dependencies && (
                          <Box mt={2}>
                            <Text fontSize="sm" fontWeight="medium">Dependencies:</Text>
                            <UnorderedList fontSize="sm" ml={4}>
                              {result.dependencies.map((dep, depIdx) => (
                                <ListItem key={depIdx}>{dep}</ListItem>
                              ))}
                            </UnorderedList>
                          </Box>
                        )}
                        {result.sql && (
                          <Box mt={2}>
                            <Text fontSize="sm" fontWeight="medium">SQL:</Text>
                            <Code display="block" p={2} borderRadius="md" fontSize="sm">
                              {result.sql}
                            </Code>
                          </Box>
                        )}
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
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
  
  const bgColor = useColorModeValue('white', 'gray.900')
  const textColor = useColorModeValue('gray.900', 'white')
  const borderColor = useColorModeValue('gray.100', 'gray.700')
  const primaryColor = 'orange.500'

  // Initialize or load conversation from ID
  useEffect(() => {
    // Clear messages when component mounts if there's no conversation ID
    if (!conversationId) {
      setMessages([]);
      setActiveConversationId(null);
      setCurrentConversationId(null);
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
      setProcessingStep("Loading conversation...");
      console.log("Fetching conversation with ID:", id);
      
      const response = await fetch(`http://localhost:8000/api/conversation/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching conversation: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("Conversation data:", data);
      
      if (!data.conversation) {
        throw new Error("No conversation data received");
      }
      
      // Extract conversation data
      const conversation = data.conversation;
      
      // Create properly formatted messages
      const formattedMessages = [];
      
      // Add user message
      if (conversation.query) {
        formattedMessages.push({
          id: `user-${id}`,
          role: 'user',
          content: conversation.query,
          timestamp: conversation.timestamp
        });
      }
      
      // Add assistant message
      if (conversation.response) {
        formattedMessages.push({
          id: `assistant-${id}`,
          role: 'assistant',
          type: 'architect',
          content: conversation.response,
          details: {
            ...conversation.technical_details,
            conversation_id: id
          },
          timestamp: conversation.timestamp
        });
      }
      
      // Update state
      setMessages(formattedMessages);
      setActiveConversationId(id);
      setCurrentConversationId(id);
      
    } catch (error) {
      console.error('Error fetching conversation:', error);
      toast({
        title: 'Error Loading Conversation',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setProcessingStep(null);
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
    navigate('/chat', { replace: true });
  };

  // Handle conversation selection
  const handleConversationSelect = (id) => {
    fetchConversation(id);
    onClose(); // Close the drawer on mobile
  };

  // Send message and get response from data architect
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
    setProcessingStep('Analyzing your question with Data Architect...');
    
    try {
      // Use the Data Architect agent endpoint
      const response = await fetch('http://localhost:8000/architect/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          query: userMessage,
          conversation_id: currentConversationId,
          thread_id: currentConversationId
        })
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Data Architect response:", data);
      
      // Create a conversation ID if needed
      const responseConversationId = data.conversation_id || currentConversationId || uuidv4();
      
      // Add the Data Architect's response to the chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        type: 'architect',
        content: data.response,
        id: prev.length,
        details: {
          conversation_id: responseConversationId,
          question_type: data.question_type,
          processing_time: data.processing_time,
          github_results: data.github_results?.results || [],
          sql_results: data.sql_results?.results || [],
          doc_results: data.doc_results?.results || [],
          dbt_results: data.dbt_results?.results || [],
          relationship_results: data.relationship_results?.results || []
        }
      }]);
      
      // Update conversation tracking info
      setCurrentConversationId(responseConversationId);
      
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

  // Clear the current chat
  const clearChat = () => {
    // Save current conversation ID before clearing
    const currentId = activeConversationId;
    
    // Clear messages and conversation ID
    setMessages([]);
    setActiveConversationId(null);
    setCurrentConversationId(null);
    
    toast({
      title: 'Chat Cleared',
      description: 'The chat has been cleared. You can find it in the history tab.',
      status: 'info',
      duration: 3000
    });
  };

  return (
    <Box minH="100vh" bg={bgColor}>
      <Container maxW="container.lg" py={8}>
        {messages.length === 0 ? (
          // Initial empty state
          <VStack spacing={8} align="center" textAlign="center" py={20}>
            <Heading 
              size="xl" 
              color={textColor}
              lineHeight="1.2"
            >
              Data Architect Assistant
            </Heading>

            <Text fontSize="lg" color="gray.600" maxW="600px">
              Ask questions about your database schemas, data models, or get recommendations for SQL optimization
            </Text>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} pt={8} w="full">
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoServer} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">Schema Design</Text>
                <Text fontSize="sm" color="gray.600">
                  "Help me optimize my database schema"
                </Text>
              </VStack>
              
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoCodeSlash} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">dbt Models</Text>
                <Text fontSize="sm" color="gray.600">
                  "Review my dbt model structure"
                </Text>
              </VStack>
              
              <VStack 
                bg={useColorModeValue('gray.50', 'gray.800')} 
                p={6}
                borderRadius="lg"
                spacing={3}
                border="1px"
                borderColor={borderColor}
              >
                <Icon as={IoAnalytics} boxSize={6} color={primaryColor} />
                <Text fontWeight="bold">Query Analysis</Text>
                <Text fontSize="sm" color="gray.600">
                  "Optimize this SQL query"
                </Text>
              </VStack>
            </SimpleGrid>
          </VStack>
        ) : (
          // Chat messages area
          <VStack spacing={4} h="full">
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
        )}

        {/* Input Area */}
        <Box 
          position="fixed"
          bottom={0}
          left={0}
          right={0}
          p={4}
          bg={bgColor}
          borderTop="1px"
          borderColor={borderColor}
          zIndex={2}
        >
          <Container maxW="container.md">
            <HStack spacing={4}>
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about database schemas, data models, or SQL queries..."
                size="lg"
                bg={useColorModeValue('white', 'gray.800')}
                borderColor={borderColor}
                _focus={{
                  borderColor: primaryColor,
                  boxShadow: `0 0 0 1px ${primaryColor}`
                }}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    sendMessage();
                  }
                }}
              />
              <Button
                colorScheme="orange"
                size="lg"
                px={8}
                isLoading={loading}
                onClick={sendMessage}
                leftIcon={<IoSend />}
              >
                Send
              </Button>
            </HStack>
          </Container>
        </Box>
      </Container>
    </Box>
  );
};

export default ChatPage; 