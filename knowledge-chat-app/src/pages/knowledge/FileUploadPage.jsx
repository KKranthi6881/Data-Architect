import { useState, useEffect } from 'react'
import {
  Box,
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
  Switch
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
  IoKey
} from 'react-icons/io5'
import { FaExternalLinkAlt } from 'react-icons/fa'

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
    const githubUrlPattern = /^https:\/\/github\.com\/[^\/]+\/[^\/]+/;
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
    <Box maxW="1200px" mx="auto" py={8} px={4}>
      <Box mb={8}>
        <Heading size="lg" mb={2}>Knowledge Sources</Heading>
        <Text color="gray.600">
          Upload documents, SQL scripts, and connect to GitHub repositories to build your knowledge base.
        </Text>
        <Alert status="info" mt={4} borderRadius="md">
          <AlertIcon />
          <Box>
            <AlertTitle>How it works</AlertTitle>
            <AlertDescription>
              Files are processed, analyzed, and stored in a vector database for semantic search. The AI can then reference this knowledge when answering your questions.
            </AlertDescription>
          </Box>
        </Alert>
      </Box>

      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={8}>
        <Box>
          <Tabs variant="enclosed" colorScheme="brand">
            <TabList>
              <Tab>SQL Scripts</Tab>
              <Tab>PDF Documents</Tab>
              <Tab>GitHub</Tab>
            </TabList>
            <TabPanels>
              <TabPanel px={0}>
                <Card>
                  <CardHeader>
                    <Heading size="md">Upload SQL Script</Heading>
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
                        />
                        <FormHelperText>
                          Upload SQL scripts to analyze database schemas and queries
                        </FormHelperText>
                      </FormControl>
                      
                      <Button
                        leftIcon={<IoCloudUpload />}
                        colorScheme="brand"
                        onClick={handleSqlUpload}
                        isDisabled={!sqlFile || isUploading}
                        isLoading={isUploading}
                        loadingText="Uploading..."
                      >
                        Upload SQL File
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
              
              <TabPanel px={0}>
                <Card>
                  <CardHeader>
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
                        />
                        <FormHelperText>
                          Upload PDF documents to extract text and include in your knowledge base
                        </FormHelperText>
                      </FormControl>
                      
                      <Button
                        leftIcon={<IoCloudUpload />}
                        colorScheme="brand"
                        onClick={handlePdfUpload}
                        isDisabled={!pdfFile || isUploading}
                        isLoading={isUploading}
                        loadingText="Uploading..."
                      >
                        Upload PDF File
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
              
              <TabPanel px={0}>
                <Card>
                  <CardHeader>
                    <Heading size="md">Connect GitHub Repository</Heading>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={4} align="stretch">
                      <FormControl>
                        <FormLabel>GitHub Repository URL</FormLabel>
                        <Input
                          placeholder="https://github.com/username/repository"
                          value={repoUrl}
                          onChange={handleRepoUrlChange}
                        />
                        <FormHelperText>
                          Enter the full URL of the GitHub repository
                        </FormHelperText>
                      </FormControl>
                      
                      <FormControl display="flex" alignItems="center">
                        <FormLabel htmlFor="use-token" mb="0">
                          Use Personal Access Token
                        </FormLabel>
                        <Switch 
                          id="use-token" 
                          isChecked={usePersonalToken}
                          onChange={handleTogglePersonalToken}
                          colorScheme="brand"
                        />
                      </FormControl>
                      
                      {usePersonalToken && (
                        <>
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
                        </>
                      )}
                      
                      <Button
                        leftIcon={<IoLogoGithub />}
                        colorScheme="brand"
                        onClick={handleGithubUpload}
                        isDisabled={!repoUrl || isUploading}
                        isLoading={isUploading}
                        loadingText="Connecting..."
                      >
                        Connect Repository
                      </Button>
                      
                      {isUploading && (
                        <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                      )}
                    </VStack>
                  </CardBody>
                </Card>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Box>
        
        <Box>
          <Heading size="md" mb={4}>Knowledge Sources</Heading>
          <Card variant="outline">
            <CardHeader py={3} px={4} bg="gray.50">
              <HStack justify="space-between">
                <Text fontWeight="medium">Source</Text>
                <Text fontWeight="medium">Status</Text>
              </HStack>
            </CardHeader>
            <CardBody p={0}>
              {isLoading ? (
                <Box p={4} textAlign="center">
                  <Spinner size="md" color="brand.500" mb={2} />
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
                  <Text color="gray.500">No knowledge sources added yet</Text>
                </Box>
              ) : (
                <VStack spacing={0} align="stretch" divider={<Divider />}>
                  {uploadedFiles.map(file => (
                    <Box key={file.id} py={3} px={4} _hover={{ bg: 'gray.50' }}>
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
                                <Text fontSize="xs" color="gray.500">
                                  {formatFileSize(file.size)}
                                </Text>
                              )}
                              <Text fontSize="xs" color="gray.500">
                                Added {new Date(file.uploadedAt).toLocaleDateString()}
                              </Text>
                            </HStack>
                          </VStack>
                        </HStack>
                        <Badge colorScheme="green">
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
            </CardBody>
          </Card>

          <Box mt={6}>
            <Heading size="md" mb={4}>Knowledge Stats</Heading>
            <SimpleGrid columns={{ base: 1, sm: 3 }} spacing={4}>
              <Card>
                <CardBody>
                  <VStack>
                    <Icon as={IoDocument} boxSize={8} color="red.500" />
                    <Text fontSize="2xl" fontWeight="bold">{uploadedFiles.filter(f => f.type === 'pdf').length}</Text>
                    <Text fontSize="sm" color="gray.500">PDF Documents</Text>
                  </VStack>
                </CardBody>
              </Card>
              <Card>
                <CardBody>
                  <VStack>
                    <Icon as={IoCodeSlash} boxSize={8} color="blue.500" />
                    <Text fontSize="2xl" fontWeight="bold">{uploadedFiles.filter(f => f.type === 'sql').length}</Text>
                    <Text fontSize="sm" color="gray.500">SQL Scripts</Text>
                  </VStack>
                </CardBody>
              </Card>
              <Card>
                <CardBody>
                  <VStack>
                    <Icon as={IoLogoGithub} boxSize={8} color="purple.500" />
                    <Text fontSize="2xl" fontWeight="bold">{uploadedFiles.filter(f => f.type === 'github').length}</Text>
                    <Text fontSize="sm" color="gray.500">GitHub Repos</Text>
                  </VStack>
                </CardBody>
              </Card>
            </SimpleGrid>
          </Box>
        </Box>
      </SimpleGrid>
    </Box>
  )
}

export default FileUploadPage 