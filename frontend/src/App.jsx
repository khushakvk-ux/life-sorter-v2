import { useState, useEffect } from 'react';
import ChatBotNew from './components/ChatBotNew';
import ChatBotNewMobile from './components/ChatBotNewMobile';
import AboutPage from './components/AboutPage';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from './context/ThemeContext';
import './App.css';

function App() {
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [currentPage, setCurrentPage] = useState('chat');

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="app">
          {currentPage === 'about' ? (
            <AboutPage onBack={() => setCurrentPage('chat')} />
          ) : isMobile ? (
            <ChatBotNewMobile onNavigate={setCurrentPage} />
          ) : (
            <ChatBotNew onNavigate={setCurrentPage} />
          )}
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
