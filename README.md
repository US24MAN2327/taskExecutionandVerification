# taskExecutionandVerification
PDF Task Execution & Verification App:


This Streamlit application which deals with PDF documents extracts text, analyzes the sentiment, creates summaries, and stores the data in Pinecone for future retrieval of information. The app has a Task Execution and Verification Module that checks its work at every stage to ensure accuracy.

How It Performs Task Execution & Verification

This feature guarantees that the AI sequentially completes its internal tasks and that the internal audit review is adequate to limit mistakes and enhance accuracy.

Users upload a PDF, then the app parses it into chunks of text.

The app fetches the information and saves it in Pinecone using Nomic embeddings.

The AI analyzes the sentiment and summaraizes the document.

Users must approve each phase before the next one.

Error Handling & Self-Correction

The system has the capability to use another model if tone evaluation or summarization is unsuccessful. This model can be Gemma 2-9B, for instance.

It performs additional attempts autonomously to obtain more trustable results.

User Feedback Mechanism

People check and confirm the tone in the analysis and brief summaries.

If required, they can seek clarification on the results.

Finally, users get to point out other mistakes, which makes the accuracy in execution of subsequent works better and more efficient.
