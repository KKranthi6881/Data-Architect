import React, { Component } from 'react';
import {
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Text,
  Code,
  Box
} from '@chakra-ui/react';

// Error boundary component to catch rendering errors
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Component Error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert status="error" variant="solid" flexDirection="column" alignItems="center" my={4} p={4} borderRadius="md">
          <AlertIcon boxSize="40px" mr={0} />
          <AlertTitle mt={4} mb={1} fontSize="lg">Component Error</AlertTitle>
          <AlertDescription maxWidth="100%">
            <Text>There was an error rendering this component:</Text>
            <Code colorScheme="red" display="block" whiteSpace="pre-wrap" overflowX="auto" p={2} my={2}>
              {this.state.error && this.state.error.toString()}
            </Code>
            {this.state.errorInfo && (
              <Box maxHeight="200px" overflow="auto">
                <Code colorScheme="gray" display="block" whiteSpace="pre-wrap" overflowX="auto" fontSize="xs" p={2}>
                  {this.state.errorInfo.componentStack}
                </Code>
              </Box>
            )}
          </AlertDescription>
        </Alert>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 