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
  Progress
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
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: "Hello! I'm your Data Architecture Assistant. How can I help you today?",
    }
  ])
  const [loading, setLoading] = useState(false)
  const [processingStep, setProcessingStep] = useState('')

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
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: data.answer,
        details: {
          code: data.analysis?.code || '',
          documentation: data.analysis?.documentation || '',
          github: data.analysis?.github || '',
          technical: data.technical_details || '',
          sources: data.sources || {}
        }
      }])

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

  return (
    <Box p={4}>
      {/* Messages */}
      <VStack spacing={4} mb={8} align="stretch">
        {messages.map((message, index) => (
          <Box 
            key={index}
            bg={message.type === 'user' ? 'blue.50' : 'gray.50'}
            p={4}
            borderRadius="md"
            alignSelf={message.type === 'user' ? 'flex-end' : 'flex-start'}
            maxW="80%"
          >
            <Text>{message.content}</Text>
            
            {/* Show details button only for assistant messages with details */}
            {message.type === 'assistant' && message.details && (
              <Box mt={4}>
                <Accordion allowToggle>
                  <AccordionItem>
                    <AccordionButton>
                      <Box flex="1" textAlign="left">
                        View Analysis Details
                      </Box>
                      <AccordionIcon />
                    </AccordionButton>
                    <AccordionPanel>
                      <VStack align="stretch" spacing={4}>
                        {message.details.code && (
                          <Box>
                            <Text fontWeight="bold">Code Analysis:</Text>
                            <Code p={2} whiteSpace="pre-wrap">
                              {message.details.code}
                            </Code>
                          </Box>
                        )}
                        {message.details.documentation && (
                          <Box>
                            <Text fontWeight="bold">Documentation Analysis:</Text>
                            <Code p={2} whiteSpace="pre-wrap">
                              {message.details.documentation}
                            </Code>
                          </Box>
                        )}
                        {message.details.github && (
                          <Box>
                            <Text fontWeight="bold">GitHub Analysis:</Text>
                            <Code p={2} whiteSpace="pre-wrap">
                              {message.details.github}
                            </Code>
                          </Box>
                        )}
                        {message.details.technical && (
                          <Box>
                            <Text fontWeight="bold">Technical Details:</Text>
                            <Code p={2} whiteSpace="pre-wrap">
                              {message.details.technical}
                            </Code>
                          </Box>
                        )}
                      </VStack>
                    </AccordionPanel>
                  </AccordionItem>
                </Accordion>
              </Box>
            )}
          </Box>
        ))}
        
        {/* Show processing step */}
        {processingStep && (
          <Box p={4} bg="gray.100" borderRadius="md">
            <Text>{processingStep}</Text>
            <Progress size="xs" isIndeterminate mt={2} />
          </Box>
        )}
      </VStack>

      {/* Input area */}
      <HStack>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about database schemas, data models, or SQL queries..."
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          disabled={loading}
        />
        <Button 
          onClick={sendMessage} 
          isLoading={loading}
          colorScheme="blue"
        >
          Send
        </Button>
      </HStack>
    </Box>
  )
}

export default ChatPage 