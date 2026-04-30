AgriPulse: Scaling Precision Agriculture with RAG & Specialized Coffee Research

I recently collaborated with Yokie Lidiantoro, S.T. as Agricultural Researcher from ITB (Institut Teknologi Bandung) to develop AgriPulse, a Decision Support System specifically engineered for the Indonesian coffee industry. We integrated 3,884+ data chunks of specialized research into a high-performance AI architecture.

<img width="1080" height="1080" alt="White   Purple Podcast Collaboration Announcement Instagram Post" src="https://github.com/user-attachments/assets/1501b4dd-f8f3-401c-963c-a9425b3a6b3c" />

https://agripulse-ai-hyteluitb.streamlit.app/

Technical Architecture:
- ChromaDB (The Knowledge Base): We vectorized thousands of research fragments—from AHP land suitability criteria to coffee-specific pathogens (Hemileia vastatrix). By using ChromaDB, the system retrieves precise technical solutions that a generic LLM wouldn't know.
- Groq & Llama 3.3: To ensure the system is "field-ready," I leveraged Groq's LPU with Llama 3.3. This provides the deep reasoning required for complex agricultural diagnosis with near-zero latency, even when processing massive context windows.
- Hybrid Data Source: The system doesn't just rely on static journals; it integrates real-time market data through a news-scraping pipeline. This allows for a dual-analysis: technical crop protection and economic market trends.
- Streamlit Deployment: The entire engine is wrapped in a Streamlit dashboard, providing a seamless interface for farmers and researchers to interact with complex data through simple natural language.
