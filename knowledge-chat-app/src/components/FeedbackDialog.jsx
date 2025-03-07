import React, { useState } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  FormControl,
  FormLabel,
  Textarea,
  Text,
  Box,
  VStack
} from '@chakra-ui/react';

const FeedbackDialog = ({ isOpen, onClose, onSubmit, message }) => {
  const [comments, setComments] = useState('');
  
  const handleSubmit = () => {
    onSubmit({
      approved: false,
      feedback_id: message?.details?.feedback_id,
      conversation_id: message?.details?.conversation_id,
      comments: comments
    });
    onClose();
  };
  
  return (
    <Modal isOpen={isOpen} onClose={onClose} size="lg">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Provide Feedback</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4} align="stretch">
            <Box>
              <Text fontWeight="bold">Current Understanding:</Text>
              <Box p={3} bg="gray.100" borderRadius="md" mt={2}>
                <Text>{message?.content}</Text>
              </Box>
            </Box>
            
            <FormControl>
              <FormLabel>What needs improvement?</FormLabel>
              <Textarea 
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                placeholder="Please explain what's incorrect or missing in the understanding..."
                rows={5}
              />
            </FormControl>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={handleSubmit} isDisabled={!comments.trim()}>
            Submit Feedback
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default FeedbackDialog; 