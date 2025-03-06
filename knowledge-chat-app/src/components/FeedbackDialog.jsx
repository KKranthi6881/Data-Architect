import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  VStack,
  Text,
  Textarea,
  Switch,
  FormControl,
  FormLabel,
  useToast
} from '@chakra-ui/react';

const FeedbackDialog = ({ isOpen, onClose, message, onFeedbackSubmit }) => {
  const [approved, setApproved] = useState(false);
  const [comments, setComments] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const toast = useToast();

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      await onFeedbackSubmit({
        conversation_id: message.details.conversation_id,
        approved,
        comments,
        suggested_changes: null
      });
    } catch (error) {
      toast({
        title: 'Error submitting feedback',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Reset state when dialog opens
  useEffect(() => {
    if (isOpen) {
      setApproved(false);
      setComments('');
    }
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Review Response</ModalHeader>
        <ModalBody>
          <VStack spacing={4} align="stretch">
            <Text fontWeight="bold">Original Question:</Text>
            <Text>{message?.details?.parsed_question?.original_question}</Text>
            
            <Text fontWeight="bold">Interpretation:</Text>
            <Text>{message?.details?.parsed_question?.rephrased_question}</Text>
            
            <Text fontWeight="bold">Business Context:</Text>
            <Text>
              Domain: {message?.details?.analysis?.business_context?.domain}
              <br />
              Focus: {message?.details?.analysis?.business_context?.primary_objective}
            </Text>
            
            <FormControl>
              <FormLabel>Approve this response?</FormLabel>
              <Switch 
                isChecked={approved} 
                onChange={(e) => setApproved(e.target.checked)}
              />
            </FormControl>
            
            <FormControl>
              <FormLabel>Comments or Suggestions:</FormLabel>
              <Textarea
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                placeholder="Enter any feedback or suggestions for improvement..."
              />
            </FormControl>
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button 
            mr={3} 
            onClick={onClose}
            isDisabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            colorScheme="blue"
            onClick={handleSubmit}
            isLoading={isSubmitting}
            loadingText="Submitting..."
          >
            Submit Feedback
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default FeedbackDialog; 