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
  AlertIcon
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

  const simulateUpload = (fileType, files) => {
    setIsUploading(true)
    setUploadProgress(0)
    
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsUploading(false)
          
          // Add to uploaded files
          if (Array.isArray(files)) {
            // Multiple files
            const newUploadedFiles = files.map(file => ({
              id: Date.now() + Math.random(), // Ensure unique IDs
              name: file.name,
              type: fileType,
              size: file.size,
              uploadedAt: new Date().toISOString()
            }))
            
            setUploadedFiles(prev => [...prev, ...newUploadedFiles])
            
            toast({
              title: 'Upload complete',
              description: `${files.length} ${fileType} files have been uploaded successfully.`,
              status: 'success',
              duration: 5000,
              isClosable: true,
            })
          } else {
            // Single file
            setUploadedFiles(prev => [
              ...prev, 
              { 
                id: Date.now(), 
                name: files.name, 
                type: fileType,
                size: files.size,
                uploadedAt: new Date().toISOString() 
              }
            ])
            
            toast({
              title: 'Upload complete',
              description: `${files.name} has been uploaded successfully.`,
              status: 'success',
              duration: 5000,
              isClosable: true,
            })
          }
          
          // Reset form
          if (fileType === 'sql') {
            setSqlFile(null)
            setSqlFiles([])
          }
          if (fileType === 'pdf') {
            setPdfFile(null)
            setPdfFiles([])
          }
          if (fileType === 'github') setRepoUrl('')
          
          return 0
        }
        return prev + 10
      })
    }, 300)
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

  const handleGithubUpload = () => {
    if (!repoUrl) return
    
    // Extract repo name from URL
    const repoName = repoUrl.split('/').pop() || 'repository'
    simulateUpload('github', { name: repoName })
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
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

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
                          placeholder="https://github.com/username/repository"
                          value={repoUrl}
                          onChange={handleRepoUrlChange}
                        />
                      </InputGroup>
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Connect to a GitHub repository to include code and documentation in your knowledge base
                      </Text>
                    </FormControl>
                    
                    <Button
                      leftIcon={<IoLogoGithub />}
                      colorScheme="purple"
                      variant="solid"
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
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
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