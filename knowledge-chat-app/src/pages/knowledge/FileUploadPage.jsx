import { useState, useEffect } from 'react'
import {
  Box,
  Container,
  Heading,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  VStack,
  HStack,
  Card,
  CardBody,
  CardHeader,
  FormControl,
  FormLabel,
  Input,
  Button,
  Progress,
  Badge,
  Icon,
  useToast,
  InputGroup,
  InputLeftAddon,
  Divider,
  SimpleGrid,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  FormHelperText,
  Link,
  Switch,
  useColorModeValue,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Flex,
  Image,
  Stat,
  StatLabel,
  StatNumber,
  StatGroup,
  StatHelpText,
  Table,
  Thead,
  Tr,
  Th,
  Tbody,
  Td
} from '@chakra-ui/react'
import { 
  IoCloudUpload, 
  IoDocument, 
  IoLogoGithub, 
  IoCodeSlash,
  IoCheckmarkCircle,
  IoAnalytics,
  IoServer,
  IoFolder,
  IoInformationCircle,
  IoWarning,
  IoKey,
  IoChevronDown,
  IoArrowForward
} from 'react-icons/io5'
import { FaExternalLinkAlt } from 'react-icons/fa'

// Import logos
import snowflakeLogo from '../../assets/snowflake.png'
import dbtLogo from '../../assets/dbt.png'

// API URL - change this to your FastAPI backend URL
const API_URL = 'http://localhost:8000';

const FileUploadPage = () => {
  const [sqlFile, setSqlFile] = useState(null)
  const [pdfFile, setPdfFile] = useState(null)
  const [repoUrl, setRepoUrl] = useState('')
  const [githubUsername, setGithubUsername] = useState('')
  const [githubToken, setGithubToken] = useState('')
  const [usePersonalToken, setUsePersonalToken] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const toast = useToast()

  // Theme colors to match homepage
  const bgColor = useColorModeValue('white', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const primaryColor = useColorModeValue('gray.800', 'gray.100');
  const secondaryColor = useColorModeValue('gray.700', 'gray.300');
  const cardBg = useColorModeValue('white', 'gray.800');
  const textSecondary = useColorModeValue('gray.600', 'gray.400');
  const accentColor = useColorModeValue('gray.900', 'gray.100');

  // Fetch uploaded files on component mount
  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  const fetchUploadedFiles = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_URL}/files/`);
      
      if (!response.ok) {
        throw new Error(`Error fetching files: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success' && data.files) {
        setUploadedFiles(data.files.map(file => ({
          id: file.name,
          name: file.name,
          type: file.type,
          size: file.size,
          uploadedAt: file.modified
        })));
      }
    } catch (error) {
      console.error('Error fetching files:', error);
      setError(error.message);
      toast({
        title: 'Error fetching files',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSqlFileChange = (e) => {
    if (e.target.files[0]) {
      setSqlFile(e.target.files[0])
    }
  }

  const handlePdfFileChange = (e) => {
    if (e.target.files[0]) {
      setPdfFile(e.target.files[0])
    }
  }

  const handleRepoUrlChange = (e) => {
    setRepoUrl(e.target.value)
  }

  const handleGithubUsernameChange = (e) => {
    setGithubUsername(e.target.value)
  }

  const handleGithubTokenChange = (e) => {
    setGithubToken(e.target.value)
  }

  const handleTogglePersonalToken = () => {
    setUsePersonalToken(!usePersonalToken)
  }

  const handleSqlUpload = async () => {
    if (!sqlFile) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      const formData = new FormData();
      formData.append('file', sqlFile);
      
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + 10;
          return newProgress >= 90 ? 90 : newProgress;
        });
      }, 300);
      
      const response = await fetch(`${API_URL}/upload/`, {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error uploading SQL file');
      }
      
      const data = await response.json();
      setUploadProgress(100);
      
      toast({
        title: 'Upload successful',
        description: `SQL file ${sqlFile.name} has been processed and added to the knowledge base.`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Refresh the file list
      fetchUploadedFiles();
      
      // Reset the file input
      setSqlFile(null);
      
    } catch (error) {
      console.error('Error uploading SQL file:', error);
      toast({
        title: 'Upload failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handlePdfUpload = async () => {
    if (!pdfFile) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      const formData = new FormData();
      formData.append('file', pdfFile);
      
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + 5;
          return newProgress >= 90 ? 90 : newProgress;
        });
      }, 300);
      
      const response = await fetch(`${API_URL}/upload/`, {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error uploading PDF file');
      }
      
      const data = await response.json();
      setUploadProgress(100);
      
      toast({
        title: 'Upload successful',
        description: `PDF file ${pdfFile.name} has been processed and added to the knowledge base.`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Refresh the file list
      fetchUploadedFiles();
      
      // Reset the file input
      setPdfFile(null);
      
    } catch (error) {
      console.error('Error uploading PDF file:', error);
      toast({
        title: 'Upload failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleGithubUpload = async () => {
    if (!repoUrl) return;
    
    // Validate GitHub URL format
    const githubUrlPattern = /^https:\/\/[^\/]+\/[^\/]+\/[^\/]+/;
    if (!githubUrlPattern.test(repoUrl)) {
      toast({
        title: 'Invalid GitHub URL',
        description: 'Please enter a valid GitHub repository URL (https://github.com/username/repository)',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + 5;
          return newProgress >= 90 ? 90 : newProgress;
        });
      }, 500);
      
      // Clean the URL by removing trailing .git if present
      const cleanRepoUrl = repoUrl.endsWith('.git') 
        ? repoUrl.slice(0, -4) 
        : repoUrl;
      
      const response = await fetch(`${API_URL}/github/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: cleanRepoUrl,
          username: githubUsername,
          token: githubToken
        }),
      });
      
      clearInterval(progressInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error connecting GitHub repository');
      }
      
      const data = await response.json();
      setUploadProgress(100);
      
      toast({
        title: 'GitHub Repository Connected',
        description: data.message || 'Repository successfully connected',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Refresh the file list
      fetchUploadedFiles();
      
      // Reset form
      setRepoUrl('');
      
    } catch (error) {
      console.error('Error connecting GitHub repository:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const getFileIcon = (type) => {
    switch (type) {
      case 'sql':
        return IoCodeSlash;
      case 'pdf':
        return IoDocument;
      case 'github':
        return IoLogoGithub;
      default:
        return IoDocument;
    }
  };

  const getFileColor = (type) => {
    switch (type) {
      case 'sql':
        return 'blue';
      case 'pdf':
        return 'red';
      case 'github':
        return 'purple';
      default:
        return 'gray';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box bg={bgColor} minH="100vh" py={8}>
      <Container maxW="container.xl">
        {/* Header with Steps */}
        <VStack spacing={6} align="stretch" mb={8}>
          <Heading size="lg" color={primaryColor}>Data Architecture Knowledge Base</Heading>
          
          {/* Stats Bar */}
          <Box 
            borderWidth="1px"
            borderColor={borderColor}
            borderRadius="lg"
            overflow="hidden"
            mb={4}
            bg={cardBg}
          >
            <Table variant="simple" size="sm">
              <Thead bg="gray.50">
                <Tr>
                  <Th>Knowledge Source</Th>
                  <Th textAlign="center">Count</Th>
                  <Th>Type</Th>
                  <Th>Status</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td>
                    <HStack spacing={2}>
                      <Icon as={IoCodeSlash} color="blue.500" />
                      <Text fontWeight="medium">SQL Scripts</Text>
                    </HStack>
                  </Td>
                  <Td textAlign="center">
                    <Badge colorScheme="blue" fontSize="md" px={2}>
                      {uploadedFiles.filter(f => f.type === 'sql').length}
                    </Badge>
                  </Td>
                  <Td>Snowflake Schemas</Td>
                  <Td>
                    {uploadedFiles.filter(f => f.type === 'sql').length > 0 ? 
                      <Badge colorScheme="green">Ready</Badge> : 
                      <Badge colorScheme="gray">Pending</Badge>
                    }
                  </Td>
                </Tr>
                <Tr>
                  <Td>
                    <HStack spacing={2}>
                      <Icon as={IoDocument} color="red.500" />
                      <Text fontWeight="medium">PDF Documents</Text>
                    </HStack>
                  </Td>
                  <Td textAlign="center">
                    <Badge colorScheme="red" fontSize="md" px={2}>
                      {uploadedFiles.filter(f => f.type === 'pdf').length}
                    </Badge>
                  </Td>
                  <Td>Business Content</Td>
                  <Td>
                    {uploadedFiles.filter(f => f.type === 'pdf').length > 0 ? 
                      <Badge colorScheme="green">Ready</Badge> : 
                      <Badge colorScheme="gray">Pending</Badge>
                    }
                  </Td>
                </Tr>
                <Tr>
                  <Td>
                    <HStack spacing={2}>
                      <Icon as={IoLogoGithub} color="purple.500" />
                      <Text fontWeight="medium">GitHub Repos</Text>
                    </HStack>
                  </Td>
                  <Td textAlign="center">
                    <Badge colorScheme="purple" fontSize="md" px={2}>
                      {uploadedFiles.filter(f => f.type === 'github').length}
                    </Badge>
                  </Td>
                  <Td>dbt Code</Td>
                  <Td>
                    {uploadedFiles.filter(f => f.type === 'github').length > 0 ? 
                      <Badge colorScheme="green">Ready</Badge> : 
                      <Badge colorScheme="gray">Pending</Badge>
                    }
                  </Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
          
          {/* Steps Card */}
          <Box mb={8}>
            <Heading size="md" mb={4} color={primaryColor}>Setup Process</Heading>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4} mb={4}>
              <HorizontalStepCard 
                number="1" 
                title="Snowflake Schema"
                icon={IoServer}
                description="Add your database schema information"
                logo={snowflakeLogo}
                instructions={[
                  "Login to your Snowflake account",
                  "Export table definitions using SHOW commands",
                  "Save the output as .SQL file",
                  "Upload the .SQL file using the form below"
                ]}
              />
              
              <HorizontalStepCard 
                number="2" 
                title="Business Context"
                icon={IoDocument}
                description="Add business knowledge and terminology"
                instructions={[
                  "Prepare PDF documents with business terms",
                  "Include metrics definitions, SQl logic and KPIs",
                  "Add relevant abbreviations and domain knowledge",
                  "Upload PDFs using the form below"
                ]}
              />
              
              <HorizontalStepCard 
                number="3" 
                title="dbt Repository" 
                icon={IoLogoGithub}
                description="Connect your dbt code repository"
                logo={dbtLogo}
                instructions={[
                  "Have your GitHub repository URL ready",
                  "Ensure it contains dbt models and transformations",
                  "Optionally prepare a personal access token",
                  "Connect using the GitHub form below"
                ]}
              />
            </SimpleGrid>
          </Box>
        </VStack>

        {/* Upload Tabs */}
        <Box mb={8}>
          <Tabs 
            variant="soft-rounded" 
            colorScheme="orange"
            bg={cardBg}
            borderRadius="lg"
            boxShadow="sm"
            p={4}
          >
            <TabList mb={4}>
              <Tab 
                _selected={{ color: 'white', bg: 'orange.500' }} 
                fontWeight="medium"
                mx={1}
              >
                <Icon as={IoCodeSlash} mr={2} />
                SQL Schemas
              </Tab>
              <Tab 
                _selected={{ color: 'white', bg: 'orange.500' }} 
                fontWeight="medium"
                mx={1}
              >
                <Icon as={IoDocument} mr={2} />
                PDF Documents
              </Tab>
              <Tab 
                _selected={{ color: 'white', bg: 'orange.500' }} 
                fontWeight="medium"
                mx={1}
              >
                <Icon as={IoLogoGithub} mr={2} />
                GitHub
              </Tab>
            </TabList>
            
            <TabPanels>
              <TabPanel px={0}>
                <Card borderRadius="lg" boxShadow="none" border="1px solid" borderColor={borderColor}>
                  <CardHeader bg="gray.50" py={3} px={4} borderTopRadius="lg">
                    <Heading size="md">Upload SQL Schema</Heading>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={4} align="stretch">
                      <FormControl>
                        <FormLabel>Select SQL File</FormLabel>
                        <Input
                          type="file"
                          accept=".sql"
                          onChange={handleSqlFileChange}
                          p={1}
                          borderColor={borderColor}
                          _hover={{ borderColor: primaryColor }}
                        />
                        <FormHelperText color={textSecondary}>
                          Upload SQL files containing schema definitions and queries for analysis
                        </FormHelperText>
                      </FormControl>
                      
                      <Button
                        leftIcon={<IoCloudUpload />}
                        colorScheme="blue"
                        onClick={handleSqlUpload}
                        isDisabled={!sqlFile || isUploading}
                        isLoading={isUploading}
                        loadingText="Uploading..."
                        size="md"
                        width="100%"
                        mt={2}
                        bg="blue.500"
                        _hover={{ bg: "blue.600" }}
                      >
                        Upload SQL File
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="orange" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
              
              <TabPanel px={0}>
                <Card borderRadius="lg" boxShadow="none" border="1px solid" borderColor={borderColor}>
                  <CardHeader bg="gray.50" py={3} px={4} borderTopRadius="lg">
                    <Heading size="md">Upload PDF Document</Heading>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={4} align="stretch">
                      <FormControl>
                        <FormLabel>Select PDF File</FormLabel>
                        <Input
                          type="file"
                          accept=".pdf"
                          onChange={handlePdfFileChange}
                          p={1}
                          borderColor={borderColor}
                          _hover={{ borderColor: primaryColor }}
                        />
                        <FormHelperText color={textSecondary}>
                          Upload PDF documents to extract text and include in your knowledge base
                        </FormHelperText>
                      </FormControl>
                      
                      <Button
                        leftIcon={<IoCloudUpload />}
                        colorScheme="red"
                        onClick={handlePdfUpload}
                        isDisabled={!pdfFile || isUploading}
                        isLoading={isUploading}
                        loadingText="Uploading..."
                        size="md"
                        width="100%"
                        mt={2}
                      >
                        Upload PDF File
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="orange" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
              
              <TabPanel px={0}>
                <Card borderRadius="lg" boxShadow="none" border="1px solid" borderColor={borderColor} mb={6}>
                  <CardHeader bg="gray.50" py={3} px={4} borderTopRadius="lg">
                    <Heading size="md">Connect GitHub Repository</Heading>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={4} align="stretch">
                      <FormControl>
                        <FormLabel>GitHub Repository URL</FormLabel>
                        <Input
                          placeholder="https://github.com/username/repository or https://github.enterprise.com/org/repo"
                          value={repoUrl}
                          onChange={handleRepoUrlChange}
                          borderColor={borderColor}
                          _hover={{ borderColor: primaryColor }}
                          _focus={{ borderColor: primaryColor, boxShadow: `0 0 0 1px ${primaryColor}` }}
                        />
                        <FormHelperText color={textSecondary}>
                          Supports standard GitHub and enterprise GitHub instances with custom domains
                        </FormHelperText>
                      </FormControl>
                      
                      <Alert status="info" size="sm" mt={2} bg="blue.50" borderRadius="md">
                        <AlertIcon color="blue.500"/>
                        <Box fontSize="sm">
                          <Text fontWeight="medium" color="blue.700">Enterprise GitHub URLs supported</Text>
                          <Text color="blue.600">
                            For enterprise GitHub instances, use the complete URL including your custom domain
                          </Text>
                        </Box>
                      </Alert>
                      
                      <FormControl display="flex" alignItems="center">
                        <FormLabel htmlFor="use-token" mb="0">
                          Use Personal Access Token
                        </FormLabel>
                        <Switch 
                          id="use-token" 
                          isChecked={usePersonalToken}
                          onChange={handleTogglePersonalToken}
                          colorScheme="orange"
                        />
                      </FormControl>
                      
                      {usePersonalToken && (
                        <VStack spacing={3} align="stretch" ml={6} mt={2}>
                          <FormControl>
                            <FormLabel>GitHub Username</FormLabel>
                            <Input
                              placeholder="Your GitHub username"
                              value={githubUsername}
                              onChange={handleGithubUsernameChange}
                            />
                          </FormControl>
                          
                          <FormControl>
                            <FormLabel>Personal Access Token</FormLabel>
                            <InputGroup>
                              <Input
                                type="password"
                                placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                                value={githubToken}
                                onChange={handleGithubTokenChange}
                              />
                            </InputGroup>
                            <FormHelperText>
                              <Link 
                                href="https://github.com/settings/tokens" 
                                isExternal 
                                color="brand.500"
                                display="inline-flex"
                                alignItems="center"
                              >
                                Create a token with repo scope
                                <Icon as={FaExternalLinkAlt} ml={1} boxSize={3} />
                              </Link>
                            </FormHelperText>
                          </FormControl>
                          
                          <Alert status="info" size="sm">
                            <AlertIcon />
                            <Text fontSize="sm">
                              A token is required for private repositories and helps avoid rate limits.
                            </Text>
                          </Alert>
                        </VStack>
                      )}
                      
                      <Button
                        leftIcon={<IoLogoGithub />}
                        colorScheme="purple"
                        onClick={handleGithubUpload}
                        isDisabled={!repoUrl || isUploading}
                        isLoading={isUploading}
                        loadingText="Connecting..."
                        size="md"
                        width="100%"
                        mt={2}
                      >
                        Connect Repository
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="orange" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>
        
        {/* Collapsible Knowledge Sources Section */}
        <Accordion allowToggle borderRadius="lg" mt={6}>
          <AccordionItem 
            border="1px solid" 
            borderColor={borderColor} 
            borderRadius="lg" 
            bg={cardBg}
            mb={4}
          >
            <AccordionButton py={4} _expanded={{ bg: 'gray.50' }}>
              <Box flex="1" textAlign="left">
                <Heading size="md">Uploaded Knowledge Sources</Heading>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4} px={4}>
              {isLoading ? (
                <Box p={4} textAlign="center">
                  <Spinner size="md" color={primaryColor} mb={2} />
                  <Text>Loading knowledge sources...</Text>
                </Box>
              ) : error ? (
                <Box p={4} textAlign="center">
                  <Alert status="error" borderRadius="md">
                    <AlertIcon />
                    <Text>{error}</Text>
                  </Alert>
                </Box>
              ) : uploadedFiles.length === 0 ? (
                <Box p={4} textAlign="center">
                  <Text color={textSecondary}>No knowledge sources added yet</Text>
                </Box>
              ) : (
                <VStack spacing={0} align="stretch" divider={<Divider />}>
                  {uploadedFiles.map(file => (
                    <Box key={file.id} py={3} px={4} _hover={{ bg: 'gray.50' }} transition="background 0.2s">
                      <HStack justify="space-between">
                        <HStack>
                          <Icon 
                            as={getFileIcon(file.type)} 
                            color={`${getFileColor(file.type)}.500`} 
                            boxSize={5} 
                          />
                          <VStack align="start" spacing={0}>
                            <Text fontWeight="medium">{file.name}</Text>
                            <HStack spacing={2}>
                              {file.size > 0 && (
                                <Text fontSize="xs" color={textSecondary}>
                                  {formatFileSize(file.size)}
                                </Text>
                              )}
                              <Text fontSize="xs" color={textSecondary}>
                                Added {new Date(file.uploadedAt).toLocaleDateString()}
                              </Text>
                            </HStack>
                          </VStack>
                        </HStack>
                        <Badge colorScheme="green" borderRadius="full" px={2} py={1}>
                          <HStack spacing={1}>
                            <Icon as={IoCheckmarkCircle} />
                            <Text>Processed</Text>
                          </HStack>
                        </Badge>
                      </HStack>
                    </Box>
                  ))}
                </VStack>
              )}
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
        
        {/* CTA Section */}

      </Container>
    </Box>
  )
}

// StepCard Component - New component for the steps
const HorizontalStepCard = ({ number, title, icon, description, logo, instructions }) => {
  return (
    <VStack 
      spacing={3} 
      p={5} 
      bg="gray.50" 
      borderRadius="md" 
      border="1px solid"
      borderColor="gray.200"
      position="relative"
      overflow="hidden"
      alignItems="stretch"
      height="100%"
    >
      <HStack spacing={3} align="center">
        <Flex
          bg="gray.200"
          color="gray.700"
          borderRadius="full"
          width="32px"
          height="32px"
          alignItems="center"
          justifyContent="center"
          fontWeight="bold"
          flexShrink={0}
        >
          {number}
        </Flex>
        <Text fontWeight="bold" fontSize="md">{title}</Text>
        <Icon as={icon} boxSize={5} color="gray.700" ml="auto" />
      </HStack>
      
      <Divider />
      
      <Text fontSize="sm" color="gray.600" mb={2}>
        {description}
      </Text>
      
      {instructions && (
        <VStack align="start" spacing={2} pl={2}>
          {instructions.map((instruction, idx) => (
            <HStack key={idx} align="start" spacing={2}>
              <Text color="gray.500" fontWeight="bold" fontSize="xs" mt={0.5}>{idx+1}.</Text>
              <Text fontSize="sm">{instruction}</Text>
            </HStack>
          ))}
        </VStack>
      )}
      
      {logo && (
        <Image 
          src={logo}
          alt={title}
          height="18px"
          alignSelf="flex-end"
          mt="auto"
          pt={1}
          opacity={0.8}
        />
      )}
    </VStack>
  );
};

// Add a new StatsCard component
const StatsCard = ({ icon, color, value, label, bg, borderColor }) => {
  return (
    <Card 
      bg={bg} 
      borderWidth="1px" 
      borderColor={borderColor} 
      borderRadius="lg"
      boxShadow="sm"
      transition="transform 0.2s, box-shadow 0.2s"
      _hover={{ transform: 'translateY(-2px)', boxShadow: 'md' }}
    >
      <CardBody>
        <VStack spacing={2}>
          <Icon as={icon} boxSize={8} color={color} />
          <Text fontSize="2xl" fontWeight="bold">{value}</Text>
          <Text fontSize="sm" color="gray.500">{label}</Text>
        </VStack>
      </CardBody>
    </Card>
  );
};

export default FileUploadPage 