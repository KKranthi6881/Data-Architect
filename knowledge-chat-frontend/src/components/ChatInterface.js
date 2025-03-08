// Update the handleMessageSubmit function to handle the feedback interface
const handleMessageSubmit = async (message) => {
  try {
    setIsLoading(true);
    setError(null);
    
    // Add user message to chat
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    
    // Send message to backend
    const response = await fetch('/chat/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message,
        conversation_id: currentConversationId,
        thread_id: currentThreadId,
        wait_for_feedback: false
      }),
    });
    
    const data = await response.json();
    
    if (response.ok) {
      // Update conversation ID and thread ID
      setCurrentConversationId(data.conversation_id);
      setCurrentThreadId(data.thread_id || currentThreadId);
      
      // Add assistant message to chat
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date().toISOString(),
        feedback_id: data.feedback_id,
        parsed_question: data.parsed_question,
        feedback_required: data.feedback_required,
        feedback_status: data.feedback_status
      };
      
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
      
      // Check if feedback is required
      if (data.feedback_required && data.feedback_id) {
        setFeedbackRequired(true);
        setCurrentFeedbackId(data.feedback_id);
        console.log("Feedback required for ID:", data.feedback_id);
      }
      
      // Fetch updated conversations
      fetchConversations();
    } else {
      setError(data.error || 'Failed to send message');
    }
  } catch (error) {
    console.error('Error sending message:', error);
    setError('Failed to send message. Please try again.');
  } finally {
    setIsLoading(false);
  }
};

// Add a FeedbackComponent to display the feedback interface
const FeedbackComponent = ({ feedbackId, onSubmit }) => {
  const [approved, setApproved] = useState(true);
  const [comments, setComments] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(feedbackId, approved, comments);
  };
  
  return (
    <div className="feedback-container">
      <h3>Please review this analysis</h3>
      <form onSubmit={handleSubmit}>
        <div className="feedback-options">
          <label>
            <input
              type="radio"
              name="feedback"
              checked={approved}
              onChange={() => setApproved(true)}
            />
            Approve
          </label>
          <label>
            <input
              type="radio"
              name="feedback"
              checked={!approved}
              onChange={() => setApproved(false)}
            />
            Needs Improvement
          </label>
        </div>
        
        {!approved && (
          <textarea
            placeholder="Please provide feedback on what needs improvement..."
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            rows={3}
          />
        )}
        
        <button type="submit" className="submit-button">
          Submit Feedback
        </button>
      </form>
    </div>
  );
};

// Add this to the component to log messages for debugging
useEffect(() => {
  console.log("Current messages:", messages);
  console.log("Feedback required:", feedbackRequired);
  console.log("Current feedback ID:", currentFeedbackId);
}, [messages, feedbackRequired, currentFeedbackId]);

// Add a new component to format the architect's response
const ArchitectResponse = ({ content, technical_details }) => {
  // Extract sections from the content
  const formatContent = () => {
    // If there are no sections, just return the content as is
    if (!technical_details || !technical_details.sections || Object.keys(technical_details.sections).length === 0) {
      return <ReactMarkdown>{content}</ReactMarkdown>;
    }
    
    // Otherwise, format the content with collapsible sections
    return (
      <div className="architect-response">
        <h3>Data Architect Solution</h3>
        
        {/* Extract and display sections */}
        {content.split('##').map((section, index) => {
          if (index === 0) return null; // Skip the first split which is empty
          
          const sectionLines = section.trim().split('\n');
          const sectionTitle = sectionLines[0].trim();
          const sectionContent = sectionLines.slice(1).join('\n').trim();
          
          return (
            <details key={index} open={index === 1}>
              <summary className="section-header">{sectionTitle}</summary>
              <div className="section-content">
                <ReactMarkdown>{sectionContent}</ReactMarkdown>
              </div>
            </details>
          );
        })}
        
        {/* Add collapsible technical details */}
        <details className="technical-details">
          <summary>Technical Details</summary>
          <div className="details-content">
            {technical_details.schema_results && technical_details.schema_results.length > 0 && (
              <div className="schema-results">
                <h4>Schema Results</h4>
                {technical_details.schema_results.map((schema, idx) => (
                  <div key={idx} className="schema-item">
                    <h5>{schema.schema_name}.{schema.table_name}</h5>
                    <p><strong>Columns:</strong> {Array.isArray(schema.columns) ? schema.columns.join(', ') : schema.columns}</p>
                    <p><strong>Relevance:</strong> {(schema.relevance_score * 10).toFixed(1)}/10</p>
                    {schema.explanation && <p><strong>Explanation:</strong> {schema.explanation}</p>}
                  </div>
                ))}
              </div>
            )}
            
            {technical_details.code_results && technical_details.code_results.length > 0 && (
              <div className="code-results">
                <h4>Code Examples</h4>
                {technical_details.code_results.map((code, idx) => (
                  <div key={idx} className="code-item">
                    <h5>{code.file_path}</h5>
                    <p><strong>Relevance:</strong> {(code.relevance_score * 10).toFixed(1)}/10</p>
                    {code.explanation && <p><strong>Explanation:</strong> {code.explanation}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>
      </div>
    );
  };
  
  return (
    <div className="formatted-response">
      {formatContent()}
    </div>
  );
};

// Update the message rendering to use the ArchitectResponse component
{messages.map((message, index) => {
  console.log(`Rendering message ${index}:`, message);
  return (
    <div key={message.id} className={`message ${message.role}`}>
      <div className="message-content">
        {message.technical_details && Object.keys(message.technical_details).length > 0 ? (
          <ArchitectResponse 
            content={message.content} 
            technical_details={message.technical_details} 
          />
        ) : (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        )}
        
        {message.role === 'assistant' && (
          <div className="message-debug" style={{fontSize: '10px', color: '#888', display: 'none'}}>
            feedback_id: {message.feedback_id || 'none'}<br/>
            feedback_required: {String(message.feedback_required)}<br/>
            feedback_status: {message.feedback_status || 'none'}
          </div>
        )}
        
        {message.role === 'assistant' && 
         message.feedback_required && 
         message.feedback_id && 
         message.feedback_status === 'pending' && (
          <FeedbackComponent
            feedbackId={message.feedback_id}
            onSubmit={handleFeedbackSubmit}
          />
        )}
      </div>
      <div className="message-timestamp">
        {formatTimestamp(message.timestamp)}
      </div>
    </div>
  );
})}

// Update the handleFeedbackSubmit function to handle architect response
const handleFeedbackSubmit = async (feedbackId, approved, comments) => {
  try {
    setIsSubmittingFeedback(true);
    setFeedbackError(null); // Clear any previous errors
    
    console.log(`Submitting feedback: ID=${feedbackId}, approved=${approved}, comments=${comments}`);
    
    const response = await fetch('/feedback/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        feedback_id: feedbackId,
        conversation_id: currentConversationId,
        approved: approved,
        comments: comments
      }),
    });
    
    const data = await response.json();
    console.log("Feedback response:", data);
    
    if (response.ok && data.status === 'success') {
      // If already processed, just refresh the conversation list
      if (data.already_processed) {
        console.log("Feedback already processed, refreshing conversations");
        fetchConversations();
        return;
      }
      
      // If we have an architect response, display it
      if (data.architect_response) {
        console.log("Received architect response:", data.architect_response);
        
        // Add the architect's response to the chat
        const architectResponse = data.architect_response.response;
        
        // Create a new message with the architect's response
        const newMessage = {
          id: uuidv4(), // Always generate a new UUID to ensure uniqueness
          role: 'assistant',
          content: architectResponse,
          timestamp: new Date().toISOString(),
          technical_details: {
            sections: data.architect_response.sections || {},
            schema_results: data.architect_response.schema_results || [],
            code_results: data.architect_response.code_results || []
          }
        };
        
        console.log("Adding new message to chat:", newMessage);
        
        // Add the message to the chat - use a callback to ensure state is updated
        setMessages(prevMessages => {
          console.log("Previous messages:", prevMessages);
          const updatedMessages = [...prevMessages, newMessage];
          console.log("Updated messages:", updatedMessages);
          return updatedMessages;
        });
        
        // Update the current conversation ID to the new one if provided
        if (data.new_conversation_id) {
          console.log("Updating conversation ID to:", data.new_conversation_id);
          setCurrentConversationId(data.new_conversation_id);
        }
      } else {
        console.log("No architect response in the data");
        // Even without an architect response, we should update the UI
        alert("Feedback submitted successfully, but no architect response was generated.");
      }
      
      // Clear feedback state
      setFeedbackRequired(false);
      setCurrentFeedbackId(null);
      
      // Refresh conversations
      fetchConversations();
    } else {
      console.error('Error submitting feedback:', data.message || 'Unknown error');
      setFeedbackError(data.message || 'Failed to submit feedback');
      alert(`Error submitting feedback: ${data.message || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Error submitting feedback:', error);
    setFeedbackError('Failed to submit feedback. Please try again.');
    alert(`Error submitting feedback: ${error.message}`);
  } finally {
    setIsSubmittingFeedback(false);
  }
};

// Add a debug component to help diagnose issues
const DebugPanel = ({ messages, feedbackRequired, currentFeedbackId }) => {
  const [showDebug, setShowDebug] = useState(false);
  
  if (!showDebug) {
    return (
      <button 
        onClick={() => setShowDebug(true)} 
        style={{ 
          position: 'fixed', 
          bottom: '10px', 
          right: '10px', 
          background: '#f0f0f0', 
          border: 'none', 
          padding: '5px 10px', 
          borderRadius: '5px',
          opacity: 0.7
        }}
      >
        Show Debug
      </button>
    );
  }
  
  return (
    <div style={{ 
      position: 'fixed', 
      bottom: '10px', 
      right: '10px', 
      background: '#f0f0f0', 
      padding: '10px', 
      borderRadius: '5px',
      maxHeight: '300px',
      overflowY: 'auto',
      width: '300px',
      zIndex: 1000
    }}>
      <button onClick={() => setShowDebug(false)} style={{ float: 'right' }}>Close</button>
      <h4>Debug Info</h4>
      <p><strong>Feedback Required:</strong> {String(feedbackRequired)}</p>
      <p><strong>Current Feedback ID:</strong> {currentFeedbackId || 'none'}</p>
      <p><strong>Messages Count:</strong> {messages.length}</p>
      <h5>Messages:</h5>
      <ul style={{ maxHeight: '150px', overflowY: 'auto' }}>
        {messages.map((msg, idx) => (
          <li key={idx}>
            <strong>{msg.role}:</strong> {msg.content.substring(0, 30)}...
            {msg.feedback_id && <span> (Feedback ID: {msg.feedback_id})</span>}
          </li>
        ))}
      </ul>
    </div>
  );
};

// Add the debug panel to your component
return (
  <div className="chat-interface">
    {/* ... existing code ... */}
    
    {/* Add the debug panel */}
    <DebugPanel 
      messages={messages} 
      feedbackRequired={feedbackRequired} 
      currentFeedbackId={currentFeedbackId} 
    />
  </div>
); 