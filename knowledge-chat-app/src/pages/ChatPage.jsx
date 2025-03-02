import { useState, useRef, useEffect } from 'react'
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
  Select
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
  IoContract
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

const ChatPage = () => {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      role: 'assistant', 
      content: 'Hello! I\'m your Data Architecture Assistant. How can I help you today?', 
      timestamp: new Date().toISOString() 
    }
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)
  const bgColor = useColorModeValue('white', 'gray.700')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  
  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])
  
  const handleSendMessage = () => {
    if (!input.trim()) return
    
    // Add user message
    const userMessage = {
      id: messages.length + 1,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)
    
    // Simulate AI response after a delay
    setTimeout(() => {
      const aiMessage = {
        id: messages.length + 2,
        role: 'assistant',
        content: `I understand you're asking about "${input}". Here's what I found in our data architecture:`,
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, aiMessage])
      setIsTyping(false)
    }, 1500)
  }
  
  return (
    <Flex h="calc(100vh - 64px)" maxW="1200px" mx="auto">
      {/* Sidebar with chat history */}
      <Box w="250px" borderRight="1px" borderColor={borderColor} p={4} display={{ base: 'none', md: 'block' }}>
        <VStack align="stretch" spacing={4}>
          <Button leftIcon={<IoAdd />} colorScheme="brand" size="sm">
            New Chat
          </Button>
          
          <Divider />
          
          <Heading size="xs" color="gray.500">RECENT CHATS</Heading>
          
          {/* Chat history items */}
          <VStack align="stretch" spacing={2}>
            <Card variant="outline" bg="brand.50" _hover={{ shadow: 'sm' }} cursor="pointer">
              <CardBody py={2} px={3}>
                <Text fontSize="sm" fontWeight="medium" noOfLines={1}>
                  Database Schema Question
                </Text>
                <Text fontSize="xs" color="gray.500">
                  Today, 10:30 AM
                </Text>
              </CardBody>
            </Card>
            
            <Card variant="outline" _hover={{ shadow: 'sm' }} cursor="pointer">
              <CardBody py={2} px={3}>
                <Text fontSize="sm" fontWeight="medium" noOfLines={1}>
                  SQL Query Optimization
                </Text>
                <Text fontSize="xs" color="gray.500">
                  Yesterday, 2:45 PM
                </Text>
              </CardBody>
            </Card>
          </VStack>
        </VStack>
      </Box>
      
      {/* Main chat area */}
      <Flex flex="1" direction="column">
        {/* Messages */}
        <VStack 
          flex="1" 
          spacing={4} 
          p={4} 
          overflowY="auto" 
          align="stretch"
        >
          {messages.map(message => (
            <HStack 
              key={message.id} 
              alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
              maxW="80%"
            >
              {message.role === 'assistant' && (
                <Avatar size="sm" icon={<IoDocumentText fontSize="1.2rem" />} bg="brand.500" />
              )}
              
              <Box 
                bg={message.role === 'user' ? 'brand.500' : bgColor}
                color={message.role === 'user' ? 'white' : 'inherit'}
                px={4} 
                py={2} 
                borderRadius="lg" 
                shadow="sm"
              >
                <Text>{message.content}</Text>
                
                {message.code && (
                  <Box mt={2} p={2} bg="gray.50" borderRadius="md" fontSize="sm">
                    <Code colorScheme="blue" whiteSpace="pre" display="block">
                      {message.code}
                    </Code>
                  </Box>
                )}
                
                <Text fontSize="xs" color={message.role === 'user' ? 'whiteAlpha.700' : 'gray.500'} textAlign="right" mt={1}>
                  {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </Text>
              </Box>
              
              {message.role === 'user' && (
                <Avatar size="sm" bg="gray.400" />
              )}
            </HStack>
          ))}
          
          {isTyping && (
            <HStack alignSelf="flex-start" maxW="80%">
              <Avatar size="sm" icon={<IoDocumentText fontSize="1.2rem" />} bg="brand.500" />
              <Box bg={bgColor} px={4} py={2} borderRadius="lg" shadow="sm">
                <Text>AI is thinking...</Text>
              </Box>
            </HStack>
          )}
          
          <div ref={messagesEndRef} />
        </VStack>
        
        {/* Input area */}
        <Box p={4} borderTop="1px" borderColor={borderColor}>
          <InputGroup>
            <Textarea
              placeholder="Ask about database schemas, data models, or SQL queries..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              pr="4.5rem"
              resize="none"
              rows={1}
              maxH="120px"
            />
            <InputRightElement width="4.5rem" h="100%">
              <IconButton
                h="1.75rem"
                size="sm"
                icon={<IoSend />}
                colorScheme="brand"
                onClick={handleSendMessage}
                isDisabled={!input.trim()}
                aria-label="Send message"
              />
            </InputRightElement>
          </InputGroup>
          <Text fontSize="xs" color="gray.500" mt={2} textAlign="center">
            Connected to Snowflake Data Warehouse
          </Text>
        </Box>
      </Flex>
    </Flex>
  )
}

export default ChatPage 