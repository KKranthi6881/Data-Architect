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
  AlertDescription,
  Link
} from '@chakra-ui/react'
import { FaGithub, FaKey, FaUser, FaLock, FaCheck, FaTimes, FaExternalLinkAlt } from 'react-icons/fa'

const GitHubConnectorPage = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionDetails, setConnectionDetails] = useState({
    username: '',
    token: '',
    usePersonalToken: true
  })
  const toast = useToast()
  const codeBg = useColorModeValue('gray.50', 'gray.700')

  const handleConnect = () => {
    // Validate required fields
    if (!connectionDetails.username || !connectionDetails.token) {
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
      
      toast({
        title: 'Connection successful',
        description: `Connected to GitHub as ${connectionDetails.username}`,
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
      description: 'GitHub connection closed',
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

  const handleSwitchChange = () => {
    setConnectionDetails({
      ...connectionDetails,
      usePersonalToken: !connectionDetails.usePersonalToken
    })
  }

  return (
    <Box maxW="1000px" mx="auto" py={4}>
      <Heading mb={6} display="flex" alignItems="center">
        <FaGithub style={{ marginRight: '12px' }} />
        GitHub Connector
      </Heading>
      
      <Text mb={6}>
        Connect to GitHub to allow the AI to access your repositories, issues, and pull requests.
      </Text>
      
      {isConnected && (
        <Alert status="success" mb={6} borderRadius="md">
          <AlertIcon />
          <Box flex="1">
            <AlertTitle>Connected to GitHub</AlertTitle>
            <AlertDescription display="block">
              Username: {connectionDetails.username}<br />
              Authentication: {connectionDetails.usePersonalToken ? 'Personal Access Token' : 'OAuth'}
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
          <Heading size="md">GitHub Connection Details</Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} align="stretch">
            <FormControl isRequired>
              <FormLabel>GitHub Username</FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaUser color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="username"
                  value={connectionDetails.username}
                  onChange={handleInputChange}
                  placeholder="your-github-username"
                  isDisabled={isConnected}
                />
              </InputGroup>
            </FormControl>
            
            <FormControl display="flex" alignItems="center">
              <Switch 
                id="use-personal-token" 
                isChecked={connectionDetails.usePersonalToken}
                onChange={handleSwitchChange}
                colorScheme="brand"
                mr={3}
                isDisabled={isConnected}
              />
              <FormLabel htmlFor="use-personal-token" mb={0}>
                Use Personal Access Token
              </FormLabel>
            </FormControl>
            
            <FormControl isRequired>
              <FormLabel>
                {connectionDetails.usePersonalToken ? 'Personal Access Token' : 'OAuth Token'}
              </FormLabel>
              <InputGroup>
                <InputLeftElement pointerEvents="none">
                  <FaKey color="gray.300" />
                </InputLeftElement>
                <Input 
                  name="token"
                  type="password"
                  value={connectionDetails.token}
                  onChange={handleInputChange}
                  placeholder="••••••••••••••••••••••••"
                  isDisabled={isConnected}
                />
              </InputGroup>
              <FormHelperText>
                {connectionDetails.usePersonalToken ? (
                  <Text>
                    Create a token with 'repo' scope. 
                    <Link 
                      href="https://github.com/settings/tokens" 
                      isExternal 
                      color="brand.500" 
                      ml={1}
                    >
                      Generate token <FaExternalLinkAlt size="0.8em" />
                    </Link>
                  </Text>
                ) : (
                  'OAuth token for GitHub API access'
                )}
              </FormHelperText>
            </FormControl>
          </VStack>
        </CardBody>
        <CardFooter>
          {!isConnected ? (
            <Button 
              colorScheme="brand" 
              leftIcon={<FaGithub />}
              onClick={handleConnect}
              isLoading={isConnecting}
              loadingText="Connecting..."
            >
              Connect to GitHub
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
            <Heading size="md">Connected GitHub Account</Heading>
          </CardHeader>
          <CardBody>
            <VStack align="stretch" spacing={4}>
              <Box bg={codeBg} p={4} borderRadius="md">
                <HStack justify="space-between">
                  <VStack align="start" spacing={1}>
                    <Heading size="sm">{connectionDetails.username}</Heading>
                    <Text fontSize="sm">
                      <Badge colorScheme="green" mr={2}>Connected</Badge>
                      Authentication: {connectionDetails.usePersonalToken ? 'Personal Access Token' : 'OAuth'}
                    </Text>
                  </VStack>
                  <FaGithub size="2em" />
                </HStack>
              </Box>
              
              <Text>
                The AI can now access your GitHub repositories, issues, and pull requests to provide more personalized assistance.
              </Text>
              
              <Alert status="info" borderRadius="md">
                <AlertIcon />
                <Text fontSize="sm">
                  Your GitHub token is securely stored and only used for API requests. We never store your token on our servers.
                </Text>
              </Alert>
            </VStack>
          </CardBody>
        </Card>
      )}
    </Box>
  )
}

export default GitHubConnectorPage 