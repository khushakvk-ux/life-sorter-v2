import { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(() => {
    // Check localStorage first - respect user's previous choice
    const savedTheme = localStorage.getItem('ikshan-theme');
    if (savedTheme) {
      return savedTheme;
    }

    // Default to dark theme
    return 'dark';
  });

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', theme);

    // Save to localStorage
    localStorage.setItem('ikshan-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => {
      if (prevTheme === 'dark') return 'blue';
      if (prevTheme === 'blue') return 'green';
      return 'dark';
    });
  };

  const value = {
    theme,
    toggleTheme,
    isDark: theme === 'dark',
    isBlue: theme === 'blue',
    isGreen: theme === 'green'
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
