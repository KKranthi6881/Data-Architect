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
    enterpriseUrl: '',
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

    if (isEnterprise && !config.enterpriseUrl) {
      toast({
        title: 'Error',
        description: 'Enterprise URL is required',
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

    try {
      // Create new configuration
      const newConfig = {
        id: Date.now(),
        ...config,
        isEnterprise,
        createdAt: new Date().toISOString(),
      };

      console.log('Sending configuration to backend:', newConfig);
      console.log('Full request payload:', [...savedConfigs, newConfig]);

      // Save to backend database
      const response = await fetch('/api/settings/github_connectors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify([...savedConfigs, newConfig]),
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response:', errorData);
        throw new Error(errorData.detail || 'Failed to save configuration');
      }

      const result = await response.json();
      console.log('Success response:', result);
      
      // Update local state
      setSavedConfigs([...savedConfigs, newConfig]);

      // Clear form
      setConfig({
        enterpriseUrl: '',
        username: '',
        token: '',
        repoUrl: '',
        isPublic: false,
      });

      toast({
        title: 'Success',
        description: 'GitHub connector configuration saved',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error saving configuration:', error);
      toast({
        title: 'Error',
        description: `Failed to save configuration: ${error.message}`,
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

                  {isEnterprise && (
                    <FormControl>
                      <FormLabel>Enterprise URL</FormLabel>
                      <Input
                        name="enterpriseUrl"
                        value={config.enterpriseUrl}
                        onChange={handleInputChange}
                        placeholder="https://github.your-company.com"
                        type="url"
                      />
                    </FormControl>
                  )}

                  <FormControl>
                    <FormLabel>Repository URL</FormLabel>
                    <Input
                      name="repoUrl"
                      value={config.repoUrl}
                      onChange={handleInputChange}
                      placeholder="https://github.com/username/repo"
                      type="url"
                    />
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
                      </FormControl>
                    </>
                  )}

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
                        
                        {savedConfig.isEnterprise && (
                          <Text fontSize="sm" color={textColor}>
                            Enterprise URL: {savedConfig.enterpriseUrl}
                          </Text>
                        )}
                        
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