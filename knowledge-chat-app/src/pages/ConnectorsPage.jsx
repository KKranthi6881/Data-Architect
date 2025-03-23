import React from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  SimpleGrid,
  Card,
  CardBody,
  CardHeader,
  Icon,
  Button,
  VStack,
  HStack,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaSnowflake, FaGithub, FaDatabase } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const ConnectorCard = ({ icon, title, description, status, to }) => {
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.600', 'gray.300');

  return (
    <Card
      bg={cardBg}
      borderWidth="1px"
      borderColor={borderColor}
      _hover={{ shadow: 'md' }}
      transition="all 0.2s"
    >
      <CardHeader>
        <HStack justify="space-between">
          <HStack>
            <Icon as={icon} boxSize={6} color="orange.500" />
            <Heading size="md">{title}</Heading>
          </HStack>
          <Badge colorScheme={status === 'active' ? 'green' : 'gray'}>
            {status}
          </Badge>
        </HStack>
      </CardHeader>
      <CardBody>
        <VStack align="start" spacing={4}>
          <Text color={textColor}>{description}</Text>
          <Button
            as={Link}
            to={to}
            colorScheme="orange"
            size="sm"
            width="full"
          >
            Configure
          </Button>
        </VStack>
      </CardBody>
    </Card>
  );
};

const ConnectorsPage = () => {
  return (
    <Box py={8}>
      <Container maxW="container.xl">
        <VStack spacing={8} align="stretch">
          <Box>
            <Heading size="lg" mb={2}>Data Connectors</Heading>
            <Text color="gray.600">
              Connect and manage your data sources to enable seamless data architecture analysis.
            </Text>
          </Box>

          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
            <ConnectorCard
              icon={FaSnowflake}
              title="Snowflake"
              description="Connect to your Snowflake data warehouse to analyze schemas, tables, and views."
              status="active"
              to="/connectors/snowflake"
            />
            <ConnectorCard
              icon={FaGithub}
              title="GitHub"
              description="Link your GitHub repositories to analyze DBT models and SQL scripts."
              status="active"
              to="/connectors/github"
            />
            <ConnectorCard
              icon={FaDatabase}
              title="Database"
              description="Connect to any SQL database to analyze its schema and relationships."
              status="inactive"
              to="/connectors/database"
            />
          </SimpleGrid>
        </VStack>
      </Container>
    </Box>
  );
};

export default ConnectorsPage; 