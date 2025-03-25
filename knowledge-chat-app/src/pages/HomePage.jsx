import { 
  Box, 
  Heading, 
  Text, 
  SimpleGrid, 
  Flex, 
  VStack, 
  Button, 
  Icon, 
  HStack, 
  Container, 
  useColorModeValue, 
  Card, 
  CardBody,
  Image,
  Tooltip,
  Link as ChakraLink,
} from '@chakra-ui/react'
import { 
  IoAnalytics, 
  IoCodeSlash, 
  IoServer, 
  IoLayers, 
  IoArrowForward, 
  IoBulb, 
  IoCheckmarkCircle,
  IoLogoGithub,
  IoCloudOutline,
  IoDesktopOutline,
  IoShieldCheckmarkOutline,
  IoLogoDiscord,
  IoMailOutline,
  IoLogoTwitter,
  IoTerminal,
  IoGitBranch,
  IoInformation,
  IoConstruct,
  IoArrowRedo
} from 'react-icons/io5'
import { Link } from 'react-router-dom'

// Import logos
import snowflakeLogo from '../assets/snowflake.png'
import dbtLogo from '../assets/dbt.png'
import gemma3Logo from '../assets/gemma3.png' // Import Gemma3 icon
import deepseekLogo from '../assets/deepseek.png' // Import Deepseek icon

const HomePage = () => {
  // Theme colors
  const bgMain = useColorModeValue('white', 'gray.900')
  const bgCard = useColorModeValue('white', 'gray.800')
  const primaryColor = 'orange.500'
  const borderColor = useColorModeValue('gray.100', 'gray.700')
  const textPrimary = useColorModeValue('gray.900', 'white')
  const textSecondary = useColorModeValue('gray.600', 'gray.400')

  return (
    <Box bg={bgMain} minH="100vh">
      {/* Navigation */}
      <Flex 
        py={4} 
        px={8} 
        borderBottom="1px" 
        borderColor={borderColor}
        justify="space-between"
        align="center"
        bg={useColorModeValue('white', 'gray.900')}
      >
        <HStack spacing={8}>
          <Heading size="lg">
            Data
            <Text as="span" color={primaryColor} fontWeight="bold">
              NEURO
            </Text>
            <Text as="span" color="gray.500" fontSize="lg">
              .AI
            </Text>
          </Heading>
        </HStack>

        <HStack spacing={4}>
          <Button
            as="a"
            href="https://discord.gg/yourserver"
            target="_blank"
            variant="ghost"
            leftIcon={<IoLogoDiscord />}
            color={textSecondary}
            _hover={{ color: primaryColor }}
          >
            Discord
          </Button>
          <Button
            as="a"
            href="https://github.com/yourusername/data-architect-agent"
            target="_blank"
            variant="ghost"
            leftIcon={<IoLogoGithub />}
            color={textSecondary}
            _hover={{ color: primaryColor }}
          >
            GitHub
          </Button>
          <Button
            as={Link}
            to="/chat"
            colorScheme="orange"
            rightIcon={<IoArrowForward />}
          >
            Try Data Architect
          </Button>
        </HStack>
      </Flex>

      {/* Hero Section */}
      <Box 
        bg={bgMain}
        pt={{ base: 16, md: 24 }} 
        pb={16}
      >
        <Container maxW="container.xl">
          <VStack spacing={6} textAlign="center">
            <Heading 
              as="h1" 
              size="2xl" 
              fontWeight="bold"
              lineHeight="1.2"
              color={textPrimary}
            >
              dbt Models{' '}
              <Text as="span" color={primaryColor}>
                Made Intelligent
              </Text>
            </Heading>
            
            <HStack spacing={4} justify="center" align="center" flexWrap="wrap">
              <Text fontSize="lg" color={textSecondary}>
                Open-source Data Architect specialized in 
              </Text>
              <Image
                src={dbtLogo}
                alt="dbt"
                height="32px"
              />
              <Text fontSize="lg" color={textSecondary}>with advanced analytics for</Text>
              <Image
                src={snowflakeLogo}
                alt="Snowflake"
                height="32px"
              />
              <Text fontSize="lg" color={textSecondary}>
                running entirely on your local machine.
              </Text>
            </HStack>

            <HStack spacing={4} pt={6}>
              <Button
                as={Link}
                to="/chat"
                colorScheme="orange"
                size="md"
                height="48px"
                px={6}
                rightIcon={<IoArrowForward />}
                _hover={{
                  transform: 'translateY(-2px)',
                  boxShadow: 'md',
                }}
              >
                Start Analyzing
              </Button>
              <Button
                as={Link}
                to="/connect-repo"
                variant="outline"
                colorScheme="orange"
                size="md"
                height="48px"
                px={6}
                leftIcon={<IoLogoGithub />}
              >
                Connect dbt Repo
              </Button>
            </HStack>
          </VStack>
        </Container>
      </Box>

      {/* How It Works Section */}
      <Box bg="gray.50" py={20}>
        <Container maxW="container.xl">
          <VStack spacing={16}>
            <VStack spacing={4} textAlign="center">
              <Heading size="xl" color={textPrimary}>How It Works</Heading>
              <Text color={textSecondary} fontSize="lg" maxW="container.md">
                100% local and secure analysis with specialized dbt models knowledge
              </Text>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 4 }} spacing={10} w="full">
              <StepCard 
                icon={IoLogoGithub}
                title="Connect dbt Repository"
                description="Link your GitHub repository with dbt models to provide context"
                stepNumber="1"
                iconBg="blue.50"
                iconColor="blue.500"
                hoverBorderColor="blue.200"
              />
              
              <StepCard 
                icon={IoAnalytics}
                title="Ask Data Questions"
                description="Get model info, lineage, development advice, or code enhancements"
                stepNumber="2"
                iconBg="purple.50"
                iconColor="purple.500"
                hoverBorderColor="purple.200"
              />
              
              <StepCard 
                icon={IoDesktopOutline}
                title="Local Processing"
                description="Ollama + Gemma3 & Deepseek 8B models run everything locally"
                stepNumber="3"
                iconBg="orange.50"
                iconColor="orange.500"
                hoverBorderColor="orange.200"
              />
              
              <StepCard 
                icon={IoShieldCheckmarkOutline}
                title="Secure Results"
                description="All your code stays on your machine - nothing sent to the cloud"
                stepNumber="4"
                iconBg="green.50"
                iconColor="green.500"
                hoverBorderColor="green.200"
              />
            </SimpleGrid>
            
            <HStack spacing={6} mt={10}>
              <HStack bg="gray.100" p={2} borderRadius="md">
                <Icon as={IoTerminal} boxSize={6} color="gray.700" />
                <Text fontWeight="bold">Ollama</Text>
              </HStack>
              <Text>+</Text>
              <HStack bg="gray.100" p={2} borderRadius="md">
                <Image src={gemma3Logo} alt="Gemma3" height="24px" />
                <Text fontWeight="bold">Gemma3</Text>
              </HStack>
              <Text>+</Text>
              <HStack bg="gray.100" p={2} borderRadius="md">
                <Image src={deepseekLogo} alt="Deepseek" height="24px" />
                <Text fontWeight="bold">Deepseek 8B</Text>
              </HStack>
            </HStack>
          </VStack>
        </Container>
      </Box>

      {/* Features Section */}
      <Box py={20}>
        <Container maxW="container.xl">
          <VStack spacing={16}>
            <VStack spacing={4} textAlign="center">
              <Heading size="xl" color={textPrimary}>Specialized in dbt Data Models</Heading>
              <Text color={textSecondary} fontSize="lg" maxW="container.md">
                Get expert guidance on model structure, lineage, and optimization with our AI assistant
              </Text>
              
              <HStack spacing={6} mt={4}>
                <Image
                  src={dbtLogo}
                  alt="dbt"
                  height="30px"
                />
                <Text>+</Text>
                <Image src={gemma3Logo} alt="Gemma3" height="30px" />
                <Text>+</Text>
                <Image src={deepseekLogo} alt="Deepseek" height="30px" />
              </HStack>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10} w="full">
              <FeatureCard 
                icon={IoInformation}
                title="Model Information"
                description="Understand model structure, purpose, and complex transformations with clear explanations"
                iconBg="blue.50"
                iconColor="blue.500"
                hoverBorderColor="blue.200"
              />
              
              <FeatureCard 
                icon={IoGitBranch}
                title="Data Lineage"
                description="Trace dependencies between models to understand data flow and impact analysis"
                iconBg="purple.50"
                iconColor="purple.500"
                hoverBorderColor="purple.200"
              />
              
              <FeatureCard 
                icon={IoArrowRedo}
                title="Code Enhancement"
                description="Get targeted improvements to existing models with precise, contextual changes"
                iconBg="orange.50"
                iconColor="orange.500"
                hoverBorderColor="orange.200"
              />

              <FeatureCard 
                icon={IoConstruct}
                title="Development Guidance"
                description="Build new models with step-by-step implementation guidance following best practices"
                iconBg="green.50"
                iconColor="green.500"
                hoverBorderColor="green.200"
              />
              
              <FeatureCard 
                icon={IoCodeSlash}
                title="dbt Optimization"
                description="Improve materializations, incremental models, and SQL patterns for better performance"
                iconBg="teal.50"
                iconColor="teal.500"
                hoverBorderColor="teal.200"
              />
              
              <FeatureCard 
                icon={IoCheckmarkCircle}
                title="Documentation Help"
                description="Generate comprehensive docs for models, columns and business definitions"
                iconBg="red.50"
                iconColor="red.500"
                hoverBorderColor="red.200"
              />
            </SimpleGrid>
          </VStack>
        </Container>
      </Box>
      
      {/* Use Cases Section */}
      <Box bg="gray.50" py={20}>
        <Container maxW="container.xl">
          <VStack spacing={16}>
            <VStack spacing={4} textAlign="center">
              <Heading size="xl" color={textPrimary}>How to Use It</Heading>
              <Text color={textSecondary} fontSize="lg" maxW="container.md">
                Ask your Data Architect agent questions about your dbt models
              </Text>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={10} w="full">
              <Card
                bg={useColorModeValue('white', 'gray.800')}
                borderWidth="1px"
                borderColor={useColorModeValue('gray.100', 'gray.700')}
                borderRadius="xl"
                overflow="hidden"
                boxShadow="md"
              >
                <CardBody p={6}>
                  <VStack align="start" spacing={4}>
                    <Flex
                      bg="blue.50"
                      color="blue.500"
                      p={3}
                      borderRadius="lg"
                    >
                      <Icon as={IoInformation} boxSize={6} />
                    </Flex>
                    <Heading size="md">Ask about model details</Heading>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Can you explain the fct_orders model and how it works?"
                      </Text>
                    </Box>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "What columns are in the dim_customers model?"
                      </Text>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card
                bg={useColorModeValue('white', 'gray.800')}
                borderWidth="1px"
                borderColor={useColorModeValue('gray.100', 'gray.700')}
                borderRadius="xl"
                overflow="hidden"
                boxShadow="md"
              >
                <CardBody p={6}>
                  <VStack align="start" spacing={4}>
                    <Flex
                      bg="purple.50"
                      color="purple.500"
                      p={3}
                      borderRadius="lg"
                    >
                      <Icon as={IoGitBranch} boxSize={6} />
                    </Flex>
                    <Heading size="md">Understand data lineage</Heading>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "What models depend on stg_orders?"
                      </Text>
                    </Box>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Show me the lineage for the order_items model"
                      </Text>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card
                bg={useColorModeValue('white', 'gray.800')}
                borderWidth="1px"
                borderColor={useColorModeValue('gray.100', 'gray.700')}
                borderRadius="xl"
                overflow="hidden"
                boxShadow="md"
              >
                <CardBody p={6}>
                  <VStack align="start" spacing={4}>
                    <Flex
                      bg="orange.50"
                      color="orange.500"
                      p={3}
                      borderRadius="lg"
                    >
                      <Icon as={IoArrowRedo} boxSize={6} />
                    </Flex>
                    <Heading size="md">Enhance existing models</Heading>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Enhance models/marts/core/fct_orders.sql to add avg_gross_item_sales_amount"
                      </Text>
                    </Box>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Help me optimize the stg_orders model for better performance"
                      </Text>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card
                bg={useColorModeValue('white', 'gray.800')}
                borderWidth="1px"
                borderColor={useColorModeValue('gray.100', 'gray.700')}
                borderRadius="xl"
                overflow="hidden"
                boxShadow="md"
              >
                <CardBody p={6}>
                  <VStack align="start" spacing={4}>
                    <Flex
                      bg="green.50"
                      color="green.500"
                      p={3}
                      borderRadius="lg"
                    >
                      <Icon as={IoConstruct} boxSize={6} />
                    </Flex>
                    <Heading size="md">Development assistance</Heading>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Create a new incremental model for daily customer metrics"
                      </Text>
                    </Box>
                    <Box bg="gray.100" p={3} borderRadius="md" w="full">
                      <Text fontFamily="mono" color="gray.800">
                        "Help me implement a fan-out table for marketing attribution"
                      </Text>
                    </Box>
                  </VStack>
                </CardBody>
              </Card>
            </SimpleGrid>
            
            <Button
              as={Link}
              to="/chat"
              colorScheme="orange"
              size="lg"
              height="56px"
              px={8}
              rightIcon={<IoArrowForward />}
              _hover={{
                transform: 'translateY(-2px)',
                boxShadow: 'lg',
              }}
            >
              Start Using Data Architect
            </Button>
          </VStack>
        </Container>
      </Box>
    </Box>
  )
}

// Step card component
const StepCard = ({ icon, title, description, stepNumber, iconBg, iconColor, hoverBorderColor }) => {
  return (
    <Card
      bg={useColorModeValue('white', 'gray.800')}
      borderWidth="1px"
      borderColor={hoverBorderColor}
      borderRadius="xl"
      position="relative"
      _hover={{ 
        transform: 'translateY(-4px)', 
        boxShadow: 'lg',
        borderColor: hoverBorderColor 
      }}
      transition="all 0.2s"
    >
      <CardBody p={6}>
        <VStack spacing={4} align="start">
          <Flex
            bg={iconBg}
            color={iconColor}
            w={12}
            h={12}
            rounded="full"
            align="center"
            justify="center"
          >
            <Icon as={icon} boxSize={6} />
          </Flex>
          <Box>
            <Text
              position="absolute"
              top={4}
              right={4}
              fontSize="sm"
              fontWeight="bold"
              color={useColorModeValue('blue.500', 'blue.400')}
            >
              Step {stepNumber}
            </Text>
            <Heading size="md" mb={2}>{title}</Heading>
            <Text color={useColorModeValue('gray.600', 'gray.400')}>
              {description}
            </Text>
          </Box>
        </VStack>
      </CardBody>
    </Card>
  )
}

// Feature card component
const FeatureCard = ({ icon, title, description, iconBg, iconColor, hoverBorderColor }) => {
  return (
    <Card
      bg={useColorModeValue('white', 'gray.800')}
      borderWidth="1px"
      borderColor={hoverBorderColor}
      borderRadius="xl"
      _hover={{ 
        transform: 'translateY(-4px)', 
        boxShadow: 'lg',
        borderColor: hoverBorderColor 
      }}
      transition="all 0.2s"
    >
      <CardBody p={6}>
        <HStack spacing={4} align="flex-start">
          <Flex
            bg={iconBg}
            color={iconColor}
            p={3}
            borderRadius="lg"
            align="center"
            justify="center"
            minW="50px"
            minH="50px"
          >
            <Icon as={icon} boxSize={6} />
          </Flex>
          <VStack align="start" spacing={2}>
            <Heading size="md">{title}</Heading>
            <Text color={useColorModeValue('gray.600', 'gray.400')}>
              {description}
            </Text>
          </VStack>
        </HStack>
      </CardBody>
    </Card>
  )
}

export default HomePage 