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
  IoTerminal
} from 'react-icons/io5'
import { Link } from 'react-router-dom'

// Import only existing logos
import snowflakeLogo from '../assets/snowflake.png'
import dbtLogo from '../assets/dbt.png'

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
              Data Architecture{' '}
              <Text as="span" color={primaryColor}>
                Made Simple
              </Text>
            </Heading>
            
            <HStack spacing={4} justify="center" align="center" flexWrap="wrap">
              <Text fontSize="lg" color={textSecondary}>
                Open-source Data Architect Agent specialized in 
              </Text>
              <Image
                src={snowflakeLogo}
                alt="Snowflake"
                height="32px"
              />
              <Text fontSize="lg" color={textSecondary}>and</Text>
              <Image
                src={dbtLogo}
                alt="dbt"
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
                Start Building
              </Button>
              <Button
                as={Link}
                to="/upload"
                variant="outline"
                colorScheme="orange"
                size="md"
                height="48px"
                px={6}
              >
                Upload Schema
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
                100% local and secure data processing with state-of-the-art open-source models
              </Text>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 4 }} spacing={10} w="full">
              <StepCard 
                icon={IoCloudOutline}
                title="Upload Schema"
                description="Share your Snowflake schemas & dbt models from GitHub to provide context"
                stepNumber="1"
                iconBg="blue.50"
                iconColor="blue.500"
                hoverBorderColor="blue.200"
              />
              
              <StepCard 
                icon={IoAnalytics}
                title="Ask Questions"
                description="Query your data architecture or request optimizations"
                stepNumber="2"
                iconBg="purple.50"
                iconColor="purple.500"
                hoverBorderColor="purple.200"
              />
              
              <StepCard 
                icon={IoDesktopOutline}
                title="Local Processing"
                description="Ollama + Llama 3.2:3B & Deepseek 8B models run everything on your Mac"
                stepNumber="3"
                iconBg="orange.50"
                iconColor="orange.500"
                hoverBorderColor="orange.200"
              />
              
              <StepCard 
                icon={IoShieldCheckmarkOutline}
                title="Secure Results"
                description="All your data stays on your machine - nothing sent to the cloud"
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
                <Icon as={IoBulb} boxSize={6} color="gray.700" />
                <Text fontWeight="bold">Llama 3.2:3B</Text>
              </HStack>
              <Text>+</Text>
              <HStack bg="gray.100" p={2} borderRadius="md">
                <Icon as={IoAnalytics} boxSize={6} color="gray.700" />
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
              <Heading size="xl" color={textPrimary}>Specialized in Data Architecture</Heading>
              <Text color={textSecondary} fontSize="lg" maxW="container.md">
                Get expert guidance on Snowflake and dbt optimization with our AI assistant
              </Text>
              
              <HStack spacing={6} mt={4}>
                <Image
                  src={snowflakeLogo}
                  alt="Snowflake"
                  height="30px"
                />
                <Text>+</Text>
                <Image
                  src={dbtLogo}
                  alt="dbt"
                  height="30px"
                />
              </HStack>
            </VStack>

            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={10} w="full">
              <FeatureCard 
                icon={IoServer}
                title="Snowflake Optimization"
                description="Get recommendations for clustering keys, materialized views, and query performance"
                iconBg="blue.50"
                iconColor="blue.500"
                hoverBorderColor="blue.200"
              />
              
              <FeatureCard 
                icon={IoCodeSlash}
                title="dbt Model Design"
                description="Optimize your dbt models with best practices for incremental models and testing"
                iconBg="purple.50"
                iconColor="purple.500"
                hoverBorderColor="purple.200"
              />
              
              <FeatureCard 
                icon={IoLayers}
                title="Schema Analysis"
                description="Understand dependencies, identify optimization opportunities, and visualize data flows"
                iconBg="orange.50"
                iconColor="orange.500"
                hoverBorderColor="orange.200"
              />

              <FeatureCard 
                icon={IoBulb}
                title="Business Context"
                description="Translate business requirements into technical implementations with clear steps"
                iconBg="green.50"
                iconColor="green.500"
                hoverBorderColor="green.200"
              />
              
              <FeatureCard 
                icon={IoAnalytics}
                title="SQL Recommendations"
                description="Get optimized SQL query suggestions for complex analytical requirements"
                iconBg="teal.50"
                iconColor="teal.500"
                hoverBorderColor="teal.200"
              />
              
              <FeatureCard 
                icon={IoCheckmarkCircle}
                title="Best Practices"
                description="Follow industry standards and benchmarks for modern data architecture"
                iconBg="red.50"
                iconColor="red.500"
                hoverBorderColor="red.200"
              />
            </SimpleGrid>
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