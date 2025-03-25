import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  HStack,
  FormControl,
  FormLabel,
  Input,
  Button,
  Switch,
  useToast,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Icon,
  Badge,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaGithub, FaBuilding, FaGlobe } from 'react-icons/fa';

const GitHubConnectorPage = () => {
  const [isEnterprise, setIsEnterprise] = useState(false);
  const [config, setConfig] = useState({
    username: '',
    token: '',
    repoUrl: '',
    isPublic: false,
  });
  const [savedConfigs, setSavedConfigs] = useState([]);
  const toast = useToast();
  
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.600', 'gray.300');

  // Load saved configurations on mount
  useEffect(() => {
    const loadConfigurations = async () => {
      try {
        const response = await fetch('/api/settings/github_connectors');
        if (!response.ok) {
          throw new Error('Failed to fetch configurations');
        }
        const data = await response.json();
        setSavedConfigs(data.configs || []);
      } catch (error) {
        console.error('Error loading configurations:', error);
        toast({
          title: 'Error',
          description: 'Failed to load saved configurations',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    };

    loadConfigurations();
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate required fields
    if (!config.repoUrl) {
      toast({
        title: 'Error',
        description: 'Repository URL is required',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!config.isPublic && (!config.username || !config.token)) {
      toast({
        title: 'Error',
        description: 'Username and token are required for private repositories',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // Check if the repository URL is valid
    if (!isValidGitHubUrl(config.repoUrl)) {
      toast({
        title: 'Error',
        description: 'Invalid repository URL format. Please enter a valid GitHub repository URL.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      // Create new configuration
      const newConfig = {
        id: Date.now(),
        ...config,
        isEnterprise,
        createdAt: new Date().toISOString(),
      };

      // Format the repository URL correctly for the backend
      let formattedUrl = newConfig.repoUrl.trim();
      
      // Ensure URL starts with https://
      if (!formattedUrl.startsWith('http://') && !formattedUrl.startsWith('https://')) {
        formattedUrl = `https://${formattedUrl}`;
      }
      
      // Remove any trailing slashes
      formattedUrl = formattedUrl.replace(/\/+$/, '');
      
      // Add .git extension for enterprise URLs if missing
      if (isEnterprise && !formattedUrl.endsWith('.git')) {
        formattedUrl = `${formattedUrl}.git`;
      }
      
      // Update the config with formatted URL
      newConfig.repoUrl = formattedUrl;

      console.log('Sending configuration to backend:', {
        ...newConfig,
        token: '[HIDDEN]' // Hide token in logs
      });
      
      // Save to backend database
      const response = await fetch('/api/settings/github_connectors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify([...savedConfigs, newConfig]),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to save configuration');
      }

      const result = await response.json();
      console.log('Success response:', result);
      
      // Update local state
      setSavedConfigs([...savedConfigs, newConfig]);

      // Show success message
      toast({
        title: 'Success',
        description: 'GitHub connection configured successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      // Clear form
      setConfig({
        username: '',
        token: '',
        repoUrl: '',
        isPublic: false,
      });
    } catch (error) {
      console.error('Error saving configuration:', error);
      toast({
        title: 'Error',
        description: error.message || 'Failed to save configuration',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleDelete = async (id) => {
    try {
      console.log('Attempting to delete configuration:', id);
      const response = await fetch(`/api/settings/github_connectors/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Delete request failed:', errorData);
        throw new Error(errorData.detail || 'Failed to delete configuration');
      }

      // Update local state
      const updatedConfigs = savedConfigs.filter(config => config.id !== id);
      setSavedConfigs(updatedConfigs);
      
      toast({
        title: 'Success',
        description: 'Configuration deleted successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error deleting configuration:', error);
      toast({
        title: 'Error',
        description: `Failed to delete configuration: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const isValidGitHubUrl = (url) => {
    try {
      // Parse the URL
      const parsedUrl = new URL(url);
      
      // Check if it's using http or https protocol
      if (parsedUrl.protocol !== 'https:' && parsedUrl.protocol !== 'http:') return false;
      
      // Remove .git extension if present for validation
      let path = parsedUrl.pathname;
      if (path.endsWith('.git')) {
        path = path.slice(0, -4); // Remove .git extension
      }
      
      // Get path parts after removing leading/trailing slashes
      const pathParts = path.replace(/^\/|\/$/g, '').split('/');
      
      // Must have at least owner/repo format
      return pathParts.length >= 2 && pathParts[0].length > 0 && pathParts[1].length > 0;
    } catch (e) {
      // Invalid URL format
      return false;
    }
  };

  return (
    <Box py={8}>
      <Container maxW="container.xl">
        <VStack spacing={8} align="stretch">
          <Box>
            <Heading size="lg" mb={2}>GitHub Connector</Heading>
            <Text color="gray.600">
              Configure GitHub repository connections for analyzing DBT models and SQL scripts.
            </Text>
          </Box>

          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
            <CardHeader>
              <HStack spacing={4}>
                <Icon as={FaGithub} boxSize={6} color="orange.500" />
                <Heading size="md">Add New Connection</Heading>
              </HStack>
            </CardHeader>
            <CardBody>
              <form onSubmit={handleSubmit}>
                <VStack spacing={6} align="stretch">
                  <FormControl display="flex" alignItems="center">
                    <FormLabel mb="0">Enterprise GitHub</FormLabel>
                    <Switch
                      isChecked={isEnterprise}
                      onChange={(e) => setIsEnterprise(e.target.checked)}
                      colorScheme="orange"
                    />
                  </FormControl>

                  <FormControl>
                    <FormLabel>Repository URL</FormLabel>
                    <Input
                      name="repoUrl"
                      value={config.repoUrl}
                      onChange={handleInputChange}
                      placeholder={isEnterprise ? 
                        "https://source.datanerd.us/dataos/dbt_core_model.git" : 
                        "https://github.com/username/repo"}
                      type="url"
                    />
                    <Text fontSize="xs" color="gray.500" mt={1}>
                      {isEnterprise ? 
                        "Enter the complete enterprise repository URL including domain and .git extension" : 
                        "Standard GitHub repository URL"}
                    </Text>
                  </FormControl>

                  <FormControl display="flex" alignItems="center">
                    <FormLabel mb="0">Public Repository</FormLabel>
                    <Switch
                      name="isPublic"
                      isChecked={config.isPublic}
                      onChange={handleInputChange}
                      colorScheme="orange"
                    />
                  </FormControl>

                  {!config.isPublic && (
                    <>
                      <FormControl>
                        <FormLabel>Username</FormLabel>
                        <Input
                          name="username"
                          value={config.username}
                          onChange={handleInputChange}
                          placeholder="GitHub username"
                        />
                        <Text fontSize="xs" color="gray.500" mt={1}>
                          Your GitHub username for authentication
                        </Text>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Personal Access Token</FormLabel>
                        <Input
                          name="token"
                          value={config.token}
                          onChange={handleInputChange}
                          type="password"
                          placeholder="GitHub personal access token"
                        />
                        <Text fontSize="xs" color="gray.500" mt={1}>
                          Token with read access to the repository
                        </Text>
                      </FormControl>
                    </>
                  )}
                  
                  <Alert status="info" borderRadius="md">
                    <AlertIcon />
                    <Box fontSize="sm">
                      <Text fontWeight="medium">For Enterprise GitHub URLs:</Text>
                      <Text>• Enter the complete URL (e.g., https://source.datanerd.us/dataos/dbt_core_model.git)</Text>
                      <Text>• Include username and access token for authentication</Text>
                      <Text>• Make sure to use the .git extension if required by your enterprise GitHub</Text>
                    </Box>
                  </Alert>

                  <Button
                    type="submit"
                    colorScheme="orange"
                    leftIcon={<FaGithub />}
                  >
                    Add Connection
                  </Button>
                </VStack>
              </form>
            </CardBody>
          </Card>

          {savedConfigs.length > 0 && (
            <Box>
              <Heading size="md" mb={4}>Saved Connections</Heading>
              <VStack spacing={4} align="stretch">
                {savedConfigs.map((savedConfig) => (
                  <Card key={savedConfig.id} bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                    <CardBody>
                      <VStack align="stretch" spacing={3}>
                        <HStack justify="space-between">
                          <HStack>
                            <Icon
                              as={savedConfig.isEnterprise ? FaBuilding : FaGlobe}
                              color="orange.500"
                            />
                            <Heading size="sm">{savedConfig.repoUrl}</Heading>
                          </HStack>
                          <Badge colorScheme={savedConfig.isPublic ? 'green' : 'blue'}>
                            {savedConfig.isPublic ? 'Public' : 'Private'}
                          </Badge>
                        </HStack>
                        
                        {!savedConfig.isPublic && (
                          <Text fontSize="sm" color={textColor}>
                            Username: {savedConfig.username}
                          </Text>
                        )}
                        
                        <HStack justify="space-between">
                          <Text fontSize="sm" color={textColor}>
                            Added: {new Date(savedConfig.createdAt).toLocaleDateString()}
                          </Text>
                          <Button
                            size="sm"
                            colorScheme="red"
                            variant="ghost"
                            onClick={() => handleDelete(savedConfig.id)}
                          >
                            Delete
                          </Button>
                        </HStack>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </VStack>
            </Box>
          )}
        </VStack>
      </Container>
    </Box>
  );
};

export default GitHubConnectorPage; 