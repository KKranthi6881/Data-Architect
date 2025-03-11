import { useState } from 'react'
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Text,
  Icon,
  useToast,
  Card,
  CardBody,
  Badge,
  Progress,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  InputGroup,
  InputLeftAddon,
  Flex,
  Checkbox,
  SimpleGrid,
  IconButton,
  Tooltip,
  Divider,
  Alert,
  AlertIcon,
  FormHelperText
} from '@chakra-ui/react'
import { 
  IoCloudUpload, 
  IoDocument, 
  IoLogoGithub, 
  IoCodeSlash,
  IoCheckmarkCircle,
  IoWarning,
  IoClose,
  IoInformationCircle,
  IoAdd
} from 'react-icons/io5'

const FileUploadComponent = () => {
  // Single file state (keeping for backward compatibility)
  const [sqlFile, setSqlFile] = useState(null)
  const [pdfFile, setPdfFile] = useState(null)
  
  // Multiple files state
  const [sqlFiles, setSqlFiles] = useState([])
  const [pdfFiles, setPdfFiles] = useState([])
  
  // Other state
  const [repoUrl, setRepoUrl] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [useMultipleFiles, setUseMultipleFiles] = useState(false)
  const [gitZipFile, setGitZipFile] = useState(null)
  const toast = useToast()

  // Single file handlers (keeping for backward compatibility)
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

  // Multiple files handlers
  const handleMultipleSqlFilesChange = (e) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files)
      setSqlFiles(filesArray)
    }
  }

  const handleMultiplePdfFilesChange = (e) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files)
      setPdfFiles(filesArray)
    }
  }

  const handleRepoUrlChange = (e) => {
    setRepoUrl(e.target.value)
  }

  const toggleMultipleFiles = () => {
    setUseMultipleFiles(!useMultipleFiles)
    // Clear file selections when toggling
    if (!useMultipleFiles) {
      setSqlFile(null)
      setPdfFile(null)
    } else {
      setSqlFiles([])
      setPdfFiles([])
    }
  }

  const removeFileFromSelection = (fileType, index) => {
    if (fileType === 'sql') {
      setSqlFiles(prev => prev.filter((_, i) => i !== index))
    } else if (fileType === 'pdf') {
      setPdfFiles(prev => prev.filter((_, i) => i !== index))
    }
  }

  const simulateUpload = (type, file) => {
    setIsUploading(true)
    setUploadProgress(0)
    
    // Create a FormData object for the file upload
    const formData = new FormData()
    
    if (type === 'sql' || type === 'pdf') {
      // Handle multiple files
      if (Array.isArray(file)) {
        file.forEach((f, index) => {
          formData.append('files', f)
        })
      } else {
        formData.append('file', file)
      }
    } else if (type === 'github') {
      formData.append('repo_url', file.name)
    } else if (type === 'git_zip') {
      formData.append('file', file.originalFile)
    }
    
    // Create a simulated progress interval
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 95) {
          clearInterval(interval)
          return 95
        }
        return prev + 5
      })
    }, 200)
    
    // Simulate an API call
    setTimeout(() => {
      clearInterval(interval)
      setUploadProgress(100)
      
      // Add the file to the uploadedFiles list
      const newFile = {
        id: Date.now().toString(),
        name: Array.isArray(file) ? `${file.length} files uploaded` : (type === 'github' ? file.name : file.name),
        type: type === 'git_zip' ? 'github' : type, // Treat git_zip as github for display
        size: Array.isArray(file) ? file.reduce((acc, f) => acc + f.size, 0) : (file.originalFile ? file.originalFile.size : 0),
        uploadedAt: new Date().toISOString()
      }
      
      setUploadedFiles(prev => [...prev, newFile])
      
      // Reset states
      setIsUploading(false)
      setSqlFile(null)
      setPdfFile(null)
      setSqlFiles([])
      setPdfFiles([])
      setRepoUrl('')
      setGitZipFile(null)
      
      toast({
        title: 'Upload successful',
        description: `${type.toUpperCase()} ${Array.isArray(file) ? 'files' : 'file'} has been processed`,
        status: 'success',
        duration: 3000,
      })
    }, 3000)
  }

  const handleSqlUpload = () => {
    if (useMultipleFiles) {
      if (sqlFiles.length === 0) return
      simulateUpload('sql', sqlFiles)
    } else {
      if (!sqlFile) return
      simulateUpload('sql', sqlFile)
    }
  }

  const handlePdfUpload = () => {
    if (useMultipleFiles) {
      if (pdfFiles.length === 0) return
      simulateUpload('pdf', pdfFiles)
    } else {
      if (!pdfFile) return
      simulateUpload('pdf', pdfFile)
    }
  }

  const isValidGitHubUrl = (url) => {
    try {
      // Parse the URL
      const parsedUrl = new URL(url)
      // Check if it's using http or https protocol
      if (parsedUrl.protocol !== 'https:' && parsedUrl.protocol !== 'http:') return false
      
      // Get path parts after removing leading/trailing slashes
      const pathParts = parsedUrl.pathname.replace(/^\/|\/$/g, '').split('/')
      
      // Must have at least owner/repo format
      return pathParts.length >= 2 && pathParts[0].length > 0 && pathParts[1].length > 0
    } catch (e) {
      // Invalid URL format
      return false
    }
  }

  const handleGithubUpload = () => {
    if (!repoUrl) {
      toast({
        title: 'Repository URL required',
        description: 'Please enter a valid GitHub repository URL',
        status: 'error',
        duration: 3000,
      })
      return
    }
    
    if (!isValidGitHubUrl(repoUrl)) {
      toast({
        title: 'Invalid GitHub URL',
        description: 'Please enter a valid GitHub URL in the format https://github.com/owner/repo or https://github.mycompany.com/owner/repo',
        status: 'error',
        duration: 5000,
      })
      return
    }
    
    // Proceed with the upload
    simulateUpload('github', { name: repoUrl })
  }

  const handleGitZipFileChange = (e) => {
    if (e.target.files[0]) {
      setGitZipFile(e.target.files[0])
    }
  }

  const handleGitZipUpload = () => {
    if (!gitZipFile) {
      toast({
        title: 'ZIP file required',
        description: 'Please select a ZIP file containing your Git repository',
        status: 'error',
        duration: 3000,
      })
      return
    }
    
    simulateUpload('git_zip', { name: gitZipFile.name, originalFile: gitZipFile })
  }

  const getFileIcon = (fileType) => {
    switch (fileType) {
      case 'sql':
        return IoCodeSlash
      case 'pdf':
        return IoDocument
      case 'github':
        return IoLogoGithub
      default:
        return IoDocument
    }
  }

  const getFileColor = (fileType) => {
    switch (fileType) {
      case 'sql':
        return 'blue'
      case 'pdf':
        return 'red'
      case 'github':
        return 'purple'
      default:
        return 'gray'
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <Box maxW="1000px" mx="auto" py={8} px={4}>
      <Tabs colorScheme="brand" variant="enclosed">
        <TabList>
          <Tab><Icon as={IoCodeSlash} mr={2} /> SQL Files</Tab>
          <Tab><Icon as={IoDocument} mr={2} /> PDF Documents</Tab>
          <Tab><Icon as={IoLogoGithub} mr={2} /> GitHub Repository</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <Flex justify="space-between" align="center">
                      <FormLabel mb={0}>Upload SQL Schema Files</FormLabel>
                      <Checkbox 
                        colorScheme="brand" 
                        isChecked={useMultipleFiles} 
                        onChange={toggleMultipleFiles}
                      >
                        Multiple files
                      </Checkbox>
                    </Flex>
                    
                    <FormControl>
                      {useMultipleFiles ? (
                        <Input
                          type="file"
                          accept=".sql"
                          onChange={handleMultipleSqlFilesChange}
                          p={1}
                          multiple
                        />
                      ) : (
                        <Input
                          type="file"
                          accept=".sql"
                          onChange={handleSqlFileChange}
                          p={1}
                        />
                      )}
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Upload .sql Schema scripts to analyze and include in your knowledge base
                      </Text>
                    </FormControl>
                    
                    {/* Display selected files */}
                    {useMultipleFiles ? (
                      sqlFiles.length > 0 && (
                        <Box mt={2}>
                          <Text fontSize="sm" fontWeight="medium" mb={2}>
                            Selected files ({sqlFiles.length}):
                          </Text>
                          <VStack align="stretch" maxH="200px" overflowY="auto" spacing={2} p={2} bg="gray.50" borderRadius="md">
                            {sqlFiles.map((file, index) => (
                              <Flex key={index} justify="space-between" align="center" p={2} bg="white" borderRadius="md" shadow="sm">
                                <HStack>
                                  <Icon as={IoCodeSlash} color="blue.500" />
                                  <Text fontSize="sm" noOfLines={1}>{file.name}</Text>
                                </HStack>
                                <HStack>
                                  <Badge colorScheme="blue">{formatFileSize(file.size)}</Badge>
                                  <IconButton
                                    icon={<IoClose />}
                                    size="xs"
                                    variant="ghost"
                                    colorScheme="red"
                                    onClick={() => removeFileFromSelection('sql', index)}
                                    aria-label="Remove file"
                                  />
                                </HStack>
                              </Flex>
                            ))}
                          </VStack>
                        </Box>
                      )
                    ) : (
                      sqlFile && (
                        <HStack>
                          <Text fontSize="sm">Selected file: {sqlFile.name}</Text>
                          <Badge colorScheme="blue">{formatFileSize(sqlFile.size)}</Badge>
                        </HStack>
                      )
                    )}
                    
                    <Button
                      leftIcon={<IoCloudUpload />}
                      colorScheme="blue"
                      variant="solid"
                      onClick={handleSqlUpload}
                      isDisabled={(useMultipleFiles ? sqlFiles.length === 0 : !sqlFile) || isUploading}
                      isLoading={isUploading}
                      loadingText="Uploading..."
                      size="md"
                      width="100%"
                      mt={2}
                    >
                      Upload {useMultipleFiles ? `SQL Files (${sqlFiles.length})` : 'SQL File'}
                    </Button>
                    
                    {isUploading && (
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>

          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <Flex justify="space-between" align="center">
                      <FormLabel mb={0}>Upload PDF Documents</FormLabel>
                      <Checkbox 
                        colorScheme="brand" 
                        isChecked={useMultipleFiles} 
                        onChange={toggleMultipleFiles}
                      >
                        Multiple files
                      </Checkbox>
                    </Flex>
                    
                    <FormControl>
                      {useMultipleFiles ? (
                        <Input
                          type="file"
                          accept=".pdf"
                          onChange={handleMultiplePdfFilesChange}
                          p={1}
                          multiple
                        />
                      ) : (
                        <Input
                          type="file"
                          accept=".pdf"
                          onChange={handlePdfFileChange}
                          p={1}
                        />
                      )}
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Upload PDF documents to extract text and include in your knowledge base
                      </Text>
                    </FormControl>
                    
                    {/* Display selected files */}
                    {useMultipleFiles ? (
                      pdfFiles.length > 0 && (
                        <Box mt={2}>
                          <Text fontSize="sm" fontWeight="medium" mb={2}>
                            Selected files ({pdfFiles.length}):
                          </Text>
                          <VStack align="stretch" maxH="200px" overflowY="auto" spacing={2} p={2} bg="gray.50" borderRadius="md">
                            {pdfFiles.map((file, index) => (
                              <Flex key={index} justify="space-between" align="center" p={2} bg="white" borderRadius="md" shadow="sm">
                                <HStack>
                                  <Icon as={IoDocument} color="red.500" />
                                  <Text fontSize="sm" noOfLines={1}>{file.name}</Text>
                                </HStack>
                                <HStack>
                                  <Badge colorScheme="red">{formatFileSize(file.size)}</Badge>
                                  <IconButton
                                    icon={<IoClose />}
                                    size="xs"
                                    variant="ghost"
                                    colorScheme="red"
                                    onClick={() => removeFileFromSelection('pdf', index)}
                                    aria-label="Remove file"
                                  />
                                </HStack>
                              </Flex>
                            ))}
                          </VStack>
                        </Box>
                      )
                    ) : (
                      pdfFile && (
                        <HStack>
                          <Text fontSize="sm">Selected file: {pdfFile.name}</Text>
                          <Badge colorScheme="red">{formatFileSize(pdfFile.size)}</Badge>
                        </HStack>
                      )
                    )}
                    
                    <Button
                      leftIcon={<IoCloudUpload />}
                      colorScheme="red"
                      variant="solid"
                      onClick={handlePdfUpload}
                      isDisabled={(useMultipleFiles ? pdfFiles.length === 0 : !pdfFile) || isUploading}
                      isLoading={isUploading}
                      loadingText="Uploading..."
                      size="md"
                      width="100%"
                      mt={2}
                    >
                      Upload {useMultipleFiles ? `PDF Files (${pdfFiles.length})` : 'PDF Document'}
                    </Button>
                    
                    {isUploading && (
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>

          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel>GitHub Repository URL</FormLabel>
                      <InputGroup>
                        <InputLeftAddon children="URL" />
                        <Input
                          type="url"
                          placeholder="https://github.com/username/repository or enterprise Git URL"
                          value={repoUrl}
                          onChange={handleRepoUrlChange}
                        />
                      </InputGroup>
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Connect to a GitHub repository (standard or enterprise) to include code and documentation
                      </Text>
                    </FormControl>
                    
                    <Button
                      leftIcon={<IoLogoGithub />}
                      colorScheme="purple"
                      variant="solid"
                      onClick={handleGithubUpload}
                      isDisabled={!repoUrl || isUploading}
                      isLoading={isUploading && !gitZipFile}
                      loadingText="Connecting..."
                      size="md"
                      width="100%"
                      mt={2}
                    >
                      Connect Repository
                    </Button>
                  </VStack>
                </CardBody>
              </Card>
              
              <Divider />
              
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <Text fontWeight="medium" color="gray.700">
                      Or upload a ZIP file of your Git repository
                    </Text>
                    
                    <FormControl>
                      <FormLabel>Git Repository ZIP File</FormLabel>
                      <Input
                        type="file"
                        accept=".zip"
                        onChange={handleGitZipFileChange}
                        p={1}
                      />
                      <FormHelperText>
                        Export your repository as a ZIP file and upload it directly
                      </FormHelperText>
                    </FormControl>
                    
                    <Button
                      leftIcon={<IoCloudUpload />}
                      colorScheme="purple"
                      variant="outline"
                      onClick={handleGitZipUpload}
                      isDisabled={!gitZipFile || isUploading}
                      isLoading={isUploading && gitZipFile}
                      loadingText="Uploading..."
                      size="md"
                      width="100%"
                      mt={2}
                    >
                      Upload Repository ZIP
                    </Button>
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>
        </TabPanels>
      </Tabs>

      {/* Uploaded Files Section */}
      {uploadedFiles.length > 0 && (
        <Box mt={8}>
          <Text fontSize="lg" fontWeight="bold" mb={4}>Uploaded Knowledge Sources</Text>
          <VStack spacing={3} align="stretch">
            {uploadedFiles.map(file => (
              <Card key={file.id} variant="outline" _hover={{ shadow: 'sm' }}>
                <CardBody py={3} px={4}>
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
                          {file.size && (
                            <Text fontSize="xs" color="gray.500">
                              {formatFileSize(file.size)}
                            </Text>
                          )}
                          <Text fontSize="xs" color="gray.500">
                            Uploaded {new Date(file.uploadedAt).toLocaleString()}
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
                </CardBody>
              </Card>
            ))}
          </VStack>
        </Box>
      )}
    </Box>
  )
}

export default FileUploadComponent 