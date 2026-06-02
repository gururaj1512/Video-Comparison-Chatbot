import { useState, useEffect, useRef } from 'react';

interface VideoMetadata {
  title: string;
  creator_name: string;
  follower_count?: number | null;
  view_count: number | null;
  like_count: number | null;
  comment_count: number | null;
  duration_seconds: number;
  upload_date: string | null;
  hashtags: string[];
  thumbnail_url: string | null;
  video_url: string;
}

interface EngagementMetrics {
  engagement_rate: number;
  like_rate: number;
  comment_rate: number;
  virality_score: number;
}

interface VideoEntry {
  url: string;
  label: string;
  video_id: string | null;
  platform: string;
  metadata: VideoMetadata | null;
  engagement: EngagementMetrics | null;
}

interface Citation {
  video_label: string;
  chunk_id: number;
}

interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  citations?: Citation[];
  timestamp: string;
  isError?: boolean;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentStatus, setCurrentStatus] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [highlightedCard, setHighlightedCard] = useState<string | null>(null);
  const [isMobileChatOpen, setIsMobileChatOpen] = useState(false);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStatus]);

  // Load last active session if exists on load
  useEffect(() => {
    const savedSessionId = localStorage.getItem('active_session_id');
    if (savedSessionId) {
      fetchSessionData(savedSessionId);
    }
  }, []);

  const fetchSessionData = async (sid: string) => {
    try {
      const res = await fetch(`/api/sessions/${sid}`);
      if (res.ok) {
        const data = await res.json();
        setSessionId(sid);
        if (data.metadata && data.metadata.videos) {
          setVideos(data.metadata.videos);
        }
        if (data.chat_history && data.chat_history.length > 0) {
          const formattedHistory: Message[] = data.chat_history.map((msg: any, idx: number) => ({
            id: `history-${idx}`,
            sender: msg.role === 'user' ? 'user' : 'assistant',
            text: msg.content,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            citations: msg.citations || [],
          }));
          setMessages(formattedHistory);
        }
        localStorage.setItem('active_session_id', sid);
      } else {
        // Stale session, remove it
        localStorage.removeItem('active_session_id');
      }
    } catch (e) {
      console.error("Error loading session:", e);
      localStorage.removeItem('active_session_id');
    }
  };

  const handleCitationClick = (videoLabel: string) => {
    setHighlightedCard(videoLabel);
    // Find the element and scroll into view if on mobile/small screen
    const element = document.getElementById(`video-card-${videoLabel.replace(/\s+/g, '')}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    // Clear highlight after 2.5s
    setTimeout(() => {
      setHighlightedCard(null);
    }, 2500);
  };

  const handleNewComparison = async () => {
    if (sessionId) {
      // Call delete endpoint on backend to clean up resources
      try {
        await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      } catch (e) {
        console.error("Error deleting session:", e);
      }
    }
    // Reset local states
    setSessionId(null);
    setVideos([]);
    setMessages([]);
    setCurrentStatus(null);
    setIsStreaming(false);
    localStorage.removeItem('active_session_id');
  };

  const executeChat = async (questionText: string) => {
    if (!questionText.trim() || isStreaming) return;

    setPrompt('');
    setIsStreaming(true);
    setCurrentStatus('Initializing request...');

    // Add user message
    const userMsgId = `msg-${Date.now()}-user`;
    const userMessage: Message = {
      id: userMsgId,
      sender: 'user',
      text: questionText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    // Add temporary assistant typing message
    const assistantMsgId = `msg-${Date.now()}-assistant`;
    const botMessage: Message = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage, botMessage]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: questionText,
          session_id: sessionId || undefined
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Server returned an error');
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported on response.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumulatedText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const cleanLine = line.trim();
          if (!cleanLine.startsWith('data: ')) continue;
          
          const jsonStr = cleanLine.substring(6);
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'status') {
              setCurrentStatus(data.content);
            } else if (data.type === 'token') {
              // Hide status loading when actual streaming of answer begins
              setCurrentStatus(null);
              accumulatedText += data.content;
              setMessages(prev => prev.map(m => m.id === assistantMsgId ? { ...m, text: accumulatedText } : m));
            } else if (data.type === 'done') {
              setCurrentStatus(null);
              setIsStreaming(false);

              // Update message with final citations
              setMessages(prev => prev.map(m => m.id === assistantMsgId ? { 
                ...m, 
                text: accumulatedText || m.text, 
                citations: data.citations || [] 
              } : m));

              if (data.session_id) {
                setSessionId(data.session_id);
                localStorage.setItem('active_session_id', data.session_id);
              }

              if (data.session_data && data.session_data.videos) {
                setVideos(data.session_data.videos);
              }
            } else if (data.type === 'error') {
              throw new Error(data.message || 'Stream processing error');
            }
          } catch (err) {
            console.error("Error parsing stream line:", err);
          }
        }
      }
    } catch (e: any) {
      console.error("Chat streaming failed:", e);
      setCurrentStatus(null);
      setIsStreaming(false);
      
      // Update assistant message to show error
      setMessages(prev => prev.map(m => m.id === assistantMsgId ? {
        ...m,
        text: `⚠️ Error processing comparison: ${e.message || 'Unknown server error'}. Please try again.`,
        isError: true
      } : m));
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeChat(prompt);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeChat(prompt);
    }
  };

  const formatNumber = (num: any): string => {
    if (num === null || num === undefined) return 'N/A';
    const val = Number(num);
    if (isNaN(val)) return 'N/A';
    if (val >= 1_000_000) {
      return (val / 1_000_000).toFixed(1).replace(/\.0$/, '') + 'M';
    }
    if (val >= 1_000) {
      return (val / 1_000).toFixed(1).replace(/\.0$/, '') + 'K';
    }
    return val.toLocaleString();
  };

  const formatDuration = (secs: number): string => {
    if (!secs) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // Safe inline markdown & citation parser
  const parseLineContent = (lineText: string) => {
    const boldRegex = /\*\*(.*?)\*\*/g;
    const parts = [];
    let lastIndex = 0;
    let match;
    
    while ((match = boldRegex.exec(lineText)) !== null) {
      if (match.index > lastIndex) {
        parts.push(lineText.substring(lastIndex, match.index));
      }
      parts.push(<strong key={`bold-${match.index}`}>{match[1]}</strong>);
      lastIndex = boldRegex.lastIndex;
    }
    if (lastIndex < lineText.length) {
      parts.push(lineText.substring(lastIndex));
    }
    
    const content = parts.length > 0 ? parts : [lineText];
    const finalElements: React.ReactNode[] = [];
    const citationRegex = /\[(Video\s+[A-Z]\s*-\s*Chunk\s+\d+)\]/gi;
    
    content.forEach((part, partIdx) => {
      if (typeof part !== 'string') {
        finalElements.push(part);
        return;
      }
      
      let cLastIndex = 0;
      let cMatch;
      while ((cMatch = citationRegex.exec(part)) !== null) {
        if (cMatch.index > cLastIndex) {
          finalElements.push(part.substring(cLastIndex, cMatch.index));
        }
        const citationText = cMatch[0];
        const videoLabelMatch = /Video\s+([A-Z])/i.exec(citationText);
        const label = videoLabelMatch ? `Video ${videoLabelMatch[1].toUpperCase()}` : null;
        
        finalElements.push(
          <button 
            key={`cit-${partIdx}-${cMatch.index}`} 
            className="citation-badge"
            onClick={() => label && handleCitationClick(label)}
            title={`Highlight ${label}`}
          >
            {citationText}
          </button>
        );
        cLastIndex = citationRegex.lastIndex;
      }
      if (cLastIndex < part.length) {
        finalElements.push(part.substring(cLastIndex));
      }
    });
    
    return finalElements;
  };

  const renderMessageContent = (text: string) => {
    if (!text) return null;
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let currentListItems: React.ReactNode[] = [];
    
    lines.forEach((line, idx) => {
      const isListItem = line.trim().startsWith('* ') || line.trim().startsWith('- ');
      if (isListItem) {
        const itemText = line.trim().substring(2);
        currentListItems.push(
          <li key={`li-${idx}`}>
            {parseLineContent(itemText)}
          </li>
        );
      } else {
        if (currentListItems.length > 0) {
          elements.push(<ul key={`ul-${idx}`} style={{ margin: '8px 0 8px 20px', listStyleType: 'disc' }}>{currentListItems}</ul>);
          currentListItems = [];
        }
        if (line.trim() !== '') {
          elements.push(
            <p key={`p-${idx}`} style={{ marginBottom: '8px' }}>
              {parseLineContent(line)}
            </p>
          );
        } else {
          elements.push(<div key={`div-${idx}`} style={{ height: '8px' }} />);
        }
      }
    });
    
    if (currentListItems.length > 0) {
      elements.push(<ul key={`ul-end`} style={{ margin: '8px 0 8px 20px', listStyleType: 'disc' }}>{currentListItems}</ul>);
    }
    
    return elements;
  };

  // Toggle suggested prompts
  const suggestionChips = videos.length > 0 ? [
    `Why did Video A get more views than Video B?`,
    `Compare the hooks in the first 5 seconds.`,
    `Suggest improvements for Video B based on Video A.`,
    `What is the engagement rate of each video?`
  ] : [
    `Compare Anthony Gordon to barca youtube video with instagram: https://www.youtube.com/watch?v=X5nMfrRrCgo & https://www.instagram.com/p/DY1ue54Ijwt/`,
    `Why does YouTube have more views than Instagram Reels?`,
  ];

  // Helper to determine grid columns for cards
  const gridClass = videos.length <= 1 ? 'cols-1' : videos.length === 2 ? 'cols-2' : 'cols-3';

  return (
    <>
      {/* Header */}
      <header className="app-header">
        <div className="logo-container">
          <svg className="logo-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
          </svg>
          <span className="logo-text text-gradient-cyan">AG RAG Video Analyzer</span>
        </div>
        <div className="session-info">
          {sessionId && (
            <span className="session-badge" title="Deterministic session hash based on processed video URLs">
              Active: {sessionId}
            </span>
          )}
          <button className="new-btn" onClick={handleNewComparison}>
            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4"></path>
            </svg>
            New Comparison
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Panel: Comparison Metrics */}
        <section className="video-panel">
          <div className="video-panel-header">
            <span className="video-panel-title">Video Comparison Matrix</span>
            <span className="session-badge">{videos.length} Video{videos.length !== 1 ? 's' : ''} Loaded</span>
          </div>

          {videos.length === 0 ? (
            <div className="empty-state animate-fade-in">
              <svg className="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
              </svg>
              <h2 className="empty-state-title text-gradient-cyan">Compare & Contrast Social Media Videos</h2>
              <p className="empty-state-text">
                Input video links in the chat prompt below. The system automatically processes, transcribes, calculates engagement metrics, and uses RAG indexing to enable interactive comparison queries.
              </p>
              
              <div className="empty-steps-list">
                <div className="empty-step-card glass">
                  <div className="empty-step-num">1</div>
                  <div className="empty-step-content">
                    <div className="empty-step-title">Paste Video Links</div>
                    <div className="empty-step-desc">Enter YouTube or Instagram Reel links directly in your first prompt.</div>
                  </div>
                </div>
                
                <div className="empty-step-card glass">
                  <div className="empty-step-num">2</div>
                  <div className="empty-step-content">
                    <div className="empty-step-title">Auto-extraction & Analysis</div>
                    <div className="empty-step-desc">Our backend extracts captions, transcribes speech, and queries viewer metrics in parallel.</div>
                  </div>
                </div>

                <div className="empty-step-card glass">
                  <div className="empty-step-num">3</div>
                  <div className="empty-step-content">
                    <div className="empty-step-title">Chat & Learn</div>
                    <div className="empty-step-desc">Ask specific questions about hooks, viewer engagement, virality, or structural layout.</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className={`video-cards-container ${gridClass}`}>
              {videos.map((video) => {
                const isHighlighted = highlightedCard === video.label;
                const metadata = video.metadata;
                const engagement = video.engagement;
                const isYoutube = video.platform === 'youtube';

                return (
                  <article 
                    key={video.label}
                    id={`video-card-${video.label.replace(/\s+/g, '')}`}
                    className={`video-card glass-interactive ${isHighlighted ? 'highlighted' : ''}`}
                  >
                    <div className="video-card-header">
                      <div className="video-card-title-sec">
                        <span className={`video-label-badge ${isYoutube ? 'cyan' : 'purple'}`}>
                          {video.label}
                        </span>
                        <h3 className="video-card-title" title={metadata?.title || 'No Title'}>
                          {metadata?.title || 'Metadata Failed'}
                        </h3>
                        <p className="video-card-creator">
                          by @{metadata?.creator_name || 'unknown'} {metadata?.follower_count ? `(${formatNumber(metadata.follower_count)} followers)` : ''}
                        </p>
                      </div>
                      <div className={`platform-icon-badge ${isYoutube ? 'youtube' : 'instagram'}`}>
                        {isYoutube ? (
                          <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M23.498 6.163a3.003 3.003 0 00-2.11-2.107C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.388.511a3.002 3.002 0 00-2.11 2.107C0 8.053 0 12 0 12s0 3.947.502 5.837a3.003 3.003 0 002.11 2.107C4.495 20.455 12 20.455 12 20.455s7.505 0 9.388-.511a3.002 3.002 0 002.11-2.107C24 15.947 24 12 24 12s0-3.947-.502-5.837zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                          </svg>
                        ) : (
                          <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.051.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/>
                          </svg>
                        )}
                      </div>
                    </div>

                    <div className="video-preview-wrapper">
                      <div className={`video-preview-ratio ${!isYoutube ? 'reel' : ''}`}>
                        {isYoutube && video.video_id ? (
                          <iframe 
                            className="video-embed-iframe"
                            src={`https://www.youtube.com/embed/${video.video_id}`}
                            title={metadata?.title || 'YouTube video player'}
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                          ></iframe>
                        ) : metadata?.thumbnail_url ? (
                          <img 
                            src={video.platform === 'instagram' ? `/api/videos/proxy-image?url=${encodeURIComponent(metadata.thumbnail_url)}` : metadata.thumbnail_url} 
                            alt={metadata.title} 
                            className="video-thumbnail-img"
                            referrerPolicy="no-referrer"
                          />
                        ) : (
                          <div className="video-thumbnail-img" style={{ background: 'linear-gradient(135deg, #11131a 0%, #1e2230 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                            No Preview Available
                          </div>
                        )}
                        <div className="video-preview-overlay">
                          <span className="video-duration-tag">
                            {metadata ? formatDuration(metadata.duration_seconds) : '0:00'}
                          </span>
                          <a 
                            href={video.url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="video-watch-link"
                          >
                            Watch
                            <svg width="10" height="10" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                            </svg>
                          </a>
                        </div>
                      </div>
                    </div>

                    {/* Basic stats */}
                    <div className="video-metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">Views</span>
                        <span className="metric-value">{metadata ? formatNumber(metadata.view_count) : 'N/A'}</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Likes</span>
                        <span className="metric-value">{metadata ? formatNumber(metadata.like_count) : 'N/A'}</span>
                      </div>
                      <div className="metric-box">
                        <span className="metric-label">Comments</span>
                        <span className="metric-value">{metadata ? formatNumber(metadata.comment_count) : 'N/A'}</span>
                      </div>
                    </div>

                    {/* Engagement Metrics */}
                    <div className="video-engagement-section">
                      <div className="engagement-row">
                        <div className="engagement-label-row">
                          <span className="engagement-title">Virality Score</span>
                          <span className="engagement-value text-gradient-cyan">
                            {engagement ? `${engagement.virality_score}/100` : 'N/A'}
                          </span>
                        </div>
                        <div className="score-track">
                          <div 
                            className={`score-fill ${isYoutube ? 'fill-cyan' : 'fill-purple'}`}
                            style={{ width: `${engagement ? engagement.virality_score : 0}%` }}
                          />
                        </div>
                      </div>

                      <div className="engagement-row">
                        <div className="engagement-label-row">
                          <span className="engagement-title">Engagement Rate</span>
                          <span className="engagement-value">
                            {engagement ? `${engagement.engagement_rate}%` : '0%'}
                          </span>
                        </div>
                        <div className="score-track">
                          <div 
                            className={`score-fill ${isYoutube ? 'fill-cyan' : 'fill-purple'}`}
                            style={{ width: `${engagement ? Math.min(engagement.engagement_rate * 5, 100) : 0}%` }}
                            title="Visual scale normalizes low social percentage ranges"
                          />
                        </div>
                      </div>

                      <div className="sub-metrics-grid">
                        <div className="sub-metric-item">
                          <span>Like Rate:</span>
                          <span className="sub-metric-val">{engagement ? `${engagement.like_rate}%` : 'N/A'}</span>
                        </div>
                        <div className="sub-metric-item">
                          <span>Comm. Rate:</span>
                          <span className="sub-metric-val">{engagement ? `${engagement.comment_rate}%` : 'N/A'}</span>
                        </div>
                      </div>

                      {metadata && metadata.hashtags && metadata.hashtags.length > 0 && (
                        <div className="hashtags-container">
                          {metadata.hashtags.slice(0, 5).map((tag, idx) => (
                            <span key={idx} className="hashtag-chip">#{tag}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        {/* Right Panel: Chat Thread */}
        <section className={`chat-panel ${isMobileChatOpen ? 'mobile-open' : ''}`}>
          <div className="chat-header">
            <span className="chat-header-title">RAG Context Chat</span>
            <span className="session-badge">{messages.length} Messages</span>
          </div>

          <div className="chat-history">
            {messages.length === 0 ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <p>No chat history yet.</p>
                <p style={{ marginTop: '4px' }}>Provide links below to start comparing.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`chat-message-row ${message.sender}`}>
                  <div className="chat-message-bubble">
                    {message.text ? (
                      renderMessageContent(message.text)
                    ) : (
                      <span className="pulse-glow" style={{ color: 'var(--text-secondary)' }}>typing...</span>
                    )}

                    {message.citations && message.citations.length > 0 && (
                      <div className="citations-wrapper">
                        {message.citations.map((cit: any, idx) => {
                          let label = 'Video';
                          let displayStr = '';
                          
                          if (typeof cit === 'string') {
                            displayStr = cit;
                            const videoLabelMatch = /Video\s+([A-Z])/i.exec(cit);
                            label = videoLabelMatch ? `Video ${videoLabelMatch[1].toUpperCase()}` : 'Video';
                          } else if (cit && typeof cit === 'object') {
                            label = cit.video_label || 'Video';
                            displayStr = `${cit.video_label || 'Video'} - Chunk ${cit.chunk_id !== undefined ? cit.chunk_id : '?'}`;
                          }
                          
                          return (
                            <button 
                              key={idx} 
                              className="citation-badge"
                              onClick={() => handleCitationClick(label)}
                              title={`Focus on ${label}`}
                            >
                              🔖 {displayStr}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                  <span className="message-meta">{message.timestamp}</span>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          {/* SSE status messages display */}
          {currentStatus && (
            <div className="status-banner">
              <div className="status-spinner" />
              <span>{currentStatus}</span>
            </div>
          )}

          {/* Suggestions container */}
          <div className="suggestions-container">
            {suggestionChips.map((chip, idx) => (
              <button 
                key={idx} 
                className="suggestion-chip"
                onClick={() => setPrompt(chip)}
                disabled={isStreaming}
              >
                {chip.length > 45 ? `${chip.slice(0, 42)}...` : chip}
              </button>
            ))}
          </div>

          {/* Chat input panel */}
          <div className="chat-input-wrapper">
            <form onSubmit={handleFormSubmit} className="chat-input-form">
              <textarea
                ref={textareaRef}
                className="chat-textarea"
                placeholder="Ask a question or paste YouTube/Instagram Reel URLs to analyze..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isStreaming}
              />
              <button 
                type="submit" 
                className="send-btn" 
                disabled={isStreaming || !prompt.trim()}
              >
                <svg className="send-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                </svg>
              </button>
            </form>
          </div>
        </section>
      </div>

      {/* Mobile chat backdrop */}
      {isMobileChatOpen && (
        <div className="mobile-chat-backdrop visible" onClick={() => setIsMobileChatOpen(false)} />
      )}

      {/* Floating chat button — visible only on mobile via CSS */}
      <button 
        className={`chat-fab ${isMobileChatOpen ? 'active' : ''}`}
        onClick={() => setIsMobileChatOpen(prev => !prev)}
        aria-label={isMobileChatOpen ? 'Close chat' : 'Open chat'}
      >
        {isMobileChatOpen ? (
          <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        )}
        {messages.length > 0 && !isMobileChatOpen && (
          <span className="chat-fab-badge">{messages.length}</span>
        )}
      </button>
    </>
  );
}

export default App;
