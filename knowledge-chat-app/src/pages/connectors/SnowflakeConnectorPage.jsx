import { useState } from 'react'
import {
  Box,
  Heading,
  Button,
  VStack,
  Text,
  FormControl,
  FormLabel,
  Input,
  InputGroup,
  InputLeftElement,
  Card,
  CardBody,
  CardHeader,
  CardFooter,
  HStack,
  Divider,
  useToast,
  Switch,
  FormHelperText,
  Code,
  useColorModeValue,
  Badge,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react'
import { FaSnowflake, FaDatabase, FaKey, FaUser, FaServer, FaLock, FaCheck, FaTimes } from 'react-icons/fa'

const SnowflakeConnectorPage = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionDetails, setConnectionDetails] = useState({
    account: '',
    username: '',
    password: '',
    warehouse: '',
    database: '',
    schema: '',
    role: 'ACCOUNTADMIN'
  })
  const [savedConnections, setSavedConnections] = useState([])
  const toast = useToast()
  const codeBg = useColorModeValue('gray.50', 'gray.700')

  const handleConnect = () => {
    // Validate required fields
    if (!connectionDetails.account || !connectionDetails.username || !connectionDetails.password) {
      toast({
        title: 'Missing required fields',
        description: 'Please fill in all required fields.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
      return
    }
    
    setIsConnecting(true)
    
    // Simulate connection
    setTimeout(() => {
      setIsConnecting(false)
      setIsConnected(true)
      
      // Add to saved connections if not already there
      if (!savedConnections.some(conn => conn.account === connectionDetails.account && conn.username === connectionDetails.username)) {
        setSavedConnections([
          ...savedConnections,
          {
            id: Date.now().toString(),
            ...connectionDetails,
            password: '********', // Don't store actual password
            connectedAt: new Date().toISOString()
          }
        ])
      }
      
      toast({
        title: 'Connection successful',
        description: `Connected to Snowflake account ${connectionDetails.account}`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      })
    }, 2000)
  }

  const handleDisconnect = () => {
    setIsConnected(false)
    toast({
      title: 'Disconnected',
      description: 'Snowflake connection closed',
      status: 'info',
      duration: 2000,
      isClosable: true,
    })
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setConnectionDetails({
      ...connectionDetails,
      [name]: value
    })
  }

  return (
    <Box maxW="1000px" mx="auto" py={4}>
      <Heading mb={6} display="flex" alignItems="center">
        <FaSnowflake style={{ marginRight: '12px' }} />
        Snowflake Connector
      </Heading>
      
      <Text mb={6}>
        Connect to your Snowflake data warehouse to allow the AI to query your data and provide insights.
      </Text>
      
      {isConnected && (
        <Alert status="success" mb={6} borderRadius="md">
          <AlertIcon />
          <Box flex="1">
            <AlertTitle>Connected to Snowflake</AlertTitle>
            <AlertDescription display="block">
              Account: {connectionDetails.account}<br />
              Database: {connectionDetails.database || 'Default'}<br />
              Warehouse: {connectionDetails.warehouse || 'Default'}
            </AlertDescription>
          </Box>
          <Button 
            colorScheme="red" 
            variant="outline" 
            size="sm" 
            onClick={handleDisconnect}
            leftIcon={<FaTimes />}
          >
            Disconnect
          </Button>
        </Alert>
      )}
      
      <Card mb={8}>
        <CardHeader>
          <Heading size="md">Snowflake Connection Details</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} align="stretch">
            <FormControl isRequired>
              <FormLabel>Account Identifier</FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaServer color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="account"
                  value={connectionDetails.account}
                  onChange={handleInputChange}
                  placeholder="your-account-id"
                  isDisabled={isConnected}
                />
              </InputGroup>
              <FormHelperText>
                Your Snowflake account identifier (e.g., xy12345.us-east-1)
              </FormHelperText>
            </FormControl>
            
            <FormControl isRequired>
              <FormLabel>Username</FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaUser color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="username"
                  value={connectionDetails.username}
                  onChange={handleInputChange}
                  placeholder="username"
                  isDisabled={isConnected}
                />
              </InputGroup>
            </FormControl>
            
            <FormControl isRequired>
              <FormLabel>Password</FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaLock color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="password"
                  type="password"
                  value={connectionDetails.password}
                  onChange={handleInputChange}
                  placeholder="••••••••"
                  isDisabled={isConnected}
                />
              </InputGroup>
            </FormControl>
            
            <Divider />
            
            <FormControl>
              <FormLabel>Warehouse</FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaDatabase color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="warehouse"
                  value={connectionDetails.warehouse}
                  onChange={handleInputChange}
                  placeholder="COMPUTE_WH"
                  isDisabled={isConnected}
                />
              </InputGroup>
              <FormHelperText>
                The warehouse to use for queries (optional)
              </FormHelperText>
            </FormControl>
            
            <HStack spacing={4}>
              <FormControl>
                <FormLabel>Database</FormLabel>
                <Input 
                  name="database"
                  value={connectionDetails.database}
                  onChange={handleInputChange}
                  placeholder="Optional"
                  isDisabled={isConnected}
                />
              </FormControl>
              
              <FormControl>
                <FormLabel>Schema</FormLabel>
                <Input 
                  name="schema"
                  value={connectionDetails.schema}
                  onChange={handleInputChange}
                  placeholder="Optional"
                  isDisabled={isConnected}
                />
              </FormControl>
            </HStack>
            
            <FormControl>
              <FormLabel>Role</FormLabel>
              <Input 
                name="role"
                value={connectionDetails.role}
                onChange={handleInputChange}
                placeholder="ACCOUNTADMIN"
                isDisabled={isConnected}
              />
              <FormHelperText>
                The role to use for the connection (default: ACCOUNTADMIN)
              </FormHelperText>
            </FormControl>
          </VStack>
        </CardBody>
        <CardFooter>
          {!isConnected ? (
            <Button 
              colorScheme="brand" 
              leftIcon={<FaSnowflake />}
              onClick={handleConnect}
              isLoading={isConnecting}
              loadingText="Connecting..."
            >
              Connect to Snowflake
            </Button>
          ) : (
            <Button 
              colorScheme="red" 
              variant="outline"
              leftIcon={<FaTimes />}
              onClick={handleDisconnect}
            >
              Disconnect
            </Button>
          )}
        </CardFooter>
      </Card>
      
      {isConnected && (
        <Card mb={8}>
          <CardHeader>
            <Heading size="md">Sample Query</Heading>
          </CardHeader>
          <CardBody>
            <Text mb={4}>
              You can now run queries against your Snowflake database. Here's a sample query:
            </Text>
            <Box bg={codeBg} p={3} borderRadius="md" mb={4}>
              <Code display="block" whiteSpace="pre" fontFamily="monospace">
                {`SELECT * FROM ${connectionDetails.database || 'YOUR_DATABASE'}.${connectionDetails.schema || 'YOUR_SCHEMA'}.YOUR_TABLE LIMIT 10;`}
              </Code>
            </Box>
            <Text>
              The AI can now use this connection to query your Snowflake data when you ask questions about your data.
            </Text>
          </CardBody>
        </Card>
      )}
      
      {savedConnections.length > 0 && (
        <>
          <Divider mb={6} />
          <Heading size="md" mb={4}>Saved Connections</Heading>
          <VStack spacing={4} align="stretch">
            {savedConnections.map(conn => (
              <Card key={conn.id} variant="outline">
                <CardBody>
                  <HStack justify="space-between">
                    <VStack align="start" spacing={1}>
                      <Heading size="sm" display="flex" alignItems="center">
                        <FaSnowflake style={{ marginRight: '8px' }} />
                        {conn.account}
                      </Heading>
                      <Text fontSize="sm">
                        User: {conn.username} • Database: {conn.database || 'Default'}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        Last connected: {new Date(conn.connectedAt).toLocaleString()}
                      </Text>
                    </VStack>
                    <Badge colorScheme={isConnected && conn.account === connectionDetails.account ? 'green' : 'gray'}>
                      {isConnected && conn.account === connectionDetails.account ? 'Connected' : 'Disconnected'}
                    </Badge>
                  </HStack>
                </CardBody>
              </Card>
            ))}
          </VStack>
        </>
      )}
    </Box>
  )
}

export default SnowflakeConnectorPage 